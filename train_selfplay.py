"""
PONG SELF-PLAY TRAINING (v2)
============================

pip install:
    pip install stable-baselines3 gymnasium numpy

Bu script:
  - Öğrenen bir PPO modelini, kendi DONDURULMUŞ geçmiş kopyasına (frozen
    snapshot) karşı sonsuza kadar eğitir.
  - Her CHECKPOINT_STEPS timestep'te bir: checkpoint kaydeder, learner/
    opponent kazanma sayısını, win rate'i, ortalama reward'ı ve ortalama
    episode uzunluğunu yazdırır, sonra öğrenilen modelin yeni bir
    snapshot'ını alıp bunu bir sonraki rakip yapar.
  - Ctrl+C ile kesildiğinde mevcut modeli mutlaka kaydeder.
  - Render/pencere açmaz (headless).

v1'e göre düzeltilen RL hataları (kısa özet, detaylar dosya içinde
yorum olarak da var):
  1. Episode'lara zaman aşımı (max_steps) eklendi -> sonsuz rally riski
     ortadan kalktı.
  2. Kazanan/kaybeden ve reward istatistiklerini toplayan bir Callback
     eklendi -> self-play'in gerçekten çalıştığını LOGLARDAN görebiliyorsun.
  3. Mesafe tabanlı reward shaping'in ağırlığı küçültüldü ve
     kapatılabilir yapıldı -> shaping'in gol reward'ını ezip agent'ı
     "ortada dur" gibi dejenere bir davranışa sürüklemesi engellendi.
  4. Opponent'ın davranışı artık bazen stokastik (deterministic=False)
     seçilebiliyor -> agent'ın tek bir donuk rakip stiline ezberle
     (overfit) riski azaltıldı.
  5. Checkpoint / opponent değişim sırası: önce yeni opponent yüklenip
     env'e atanıyor, eski opponent ondan SONRA siliniyor.
  6. Opponent'ın gerçekten PPO modeli olduğunu ve maç yaptığını
     doğrulayan açık loglar eklendi (yüklenen dosya yolu, ilk birkaç
     opponent action'ının dağılımı).
  7. En son checkpoint'i kolayca bulmak için 'pong_selfplay_latest.zip'
     her checkpoint'te de ayrıca güncelleniyor (play_vs_ai.py bunu okur).
"""

import os
import random
import time
from collections import deque

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================
# AYARLAR
# ============================================================

# Sıfırdan eğitim: eski model kullanılmıyor.
FRESH_START = True
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

CHECKPOINT_STEPS = 250_000

# Sıfırdan başlarken ilk pencere(ler) boyunca rakip YOK (paddle sabit durur).
# Amaç: learner en temel "topu takip et / çarp / gol at" becerisini
# tamamen random bir modele karşı değil, sabit bir hedefe karşı öğrensin.
# Bu bittikten sonra normal self-play (learner'ın kendi geçmiş snapshot'ına
# karşı oynaması) başlıyor.
WARMUP_STEPS = 250_000

WIDTH = 800
HEIGHT = 500

PADDLE_W = 15
PADDLE_H = 80

PADDLE_SPEED = 7
BALL_SPEED = 6

MAX_EPISODE_STEPS = 3000       # zaman aşımı: rally'ler sonsuza gitmesin
DISTANCE_SHAPING_WEIGHT = 0.002  # eskiden 0.01 idi; gol reward'ını ezmesin
USE_DISTANCE_SHAPING = True

# Rakip bazen stokastik oynasın ki öğrenci tek bir donuk stile ezberlemesin.
OPPONENT_STOCHASTIC_PROB = 0.15


# ============================================================
# SELF-PLAY PONG
# ============================================================

class SelfPlayPong(gym.Env):

    def __init__(self):
        super().__init__()

        # 0 = yukarı, 1 = dur, 2 = aşağı
        self.action_space = spaces.Discrete(3)

        # ball_x, ball_y, ball_dx, ball_dy, learner_paddle_y, opponent_paddle_y
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        self.opponent = None
        self.steps = 0

        self.reset()

    def set_opponent(self, opponent):
        self.opponent = opponent

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ball_x = WIDTH / 2
        self.ball_y = HEIGHT / 2

        self.ball_dx = random.choice([-1, 1]) * BALL_SPEED
        self.ball_dy = random.choice([-1, 1]) * BALL_SPEED

        self.learner_y = HEIGHT / 2
        self.opponent_y = HEIGHT / 2

        self.previous_distance = abs(self.ball_y - self.learner_y)
        self.steps = 0

        return self.get_learner_obs(), {}

    # ========================================================
    # OBSERVATIONS
    # ========================================================

    def get_learner_obs(self):
        return np.array([
            self.ball_x / (WIDTH / 2) - 1,
            self.ball_y / (HEIGHT / 2) - 1,
            self.ball_dx / BALL_SPEED,
            self.ball_dy / BALL_SPEED,
            self.learner_y / (HEIGHT / 2) - 1,
            self.opponent_y / (HEIGHT / 2) - 1
        ], dtype=np.float32)

    def get_opponent_obs(self):
        # Eski model hep SOL tarafta oynayacak şekilde eğitildiği için,
        # rakip sağ tarafta oynarken oyunu aynalıyoruz. Böylece model
        # solunu hep "kendi tarafı" sanıyor.
        return np.array([
            (WIDTH - self.ball_x) / (WIDTH / 2) - 1,
            self.ball_y / (HEIGHT / 2) - 1,
            -self.ball_dx / BALL_SPEED,
            self.ball_dy / BALL_SPEED,
            self.opponent_y / (HEIGHT / 2) - 1,
            self.learner_y / (HEIGHT / 2) - 1
        ], dtype=np.float32)

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        winner = None  # "learner" / "opponent" / None (zaman aşımı)

        self.steps += 1

        # ---- LEARNER ACTION ----
        if int(action) == 0:
            self.learner_y -= PADDLE_SPEED
        elif int(action) == 2:
            self.learner_y += PADDLE_SPEED

        self.learner_y = np.clip(self.learner_y, PADDLE_H / 2, HEIGHT - PADDLE_H / 2)

        # ---- OPPONENT ACTION (gerçek PPO modeli) ----
        if self.opponent is not None:
            opponent_obs = self.get_opponent_obs()

            # Çoğunlukla deterministic (en iyi bildiği hamle), ara sıra
            # stokastik -> davranış çeşitliliği, ezber riskini azaltır.
            use_deterministic = random.random() > OPPONENT_STOCHASTIC_PROB
            opponent_action, _ = self.opponent.predict(
                opponent_obs,
                deterministic=use_deterministic
            )
            opponent_action = int(opponent_action)

            if opponent_action == 0:
                self.opponent_y -= PADDLE_SPEED
            elif opponent_action == 2:
                self.opponent_y += PADDLE_SPEED

        self.opponent_y = np.clip(self.opponent_y, PADDLE_H / 2, HEIGHT - PADDLE_H / 2)

        # ---- BALL ----
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_dy *= -1

        if self.ball_y >= HEIGHT:
            self.ball_y = HEIGHT
            self.ball_dy *= -1

        # ---- LEARNER PADDLE ----
        learner_x = 40

        if (
            self.ball_dx < 0
            and learner_x <= self.ball_x <= learner_x + PADDLE_W
            and abs(self.ball_y - self.learner_y) <= PADDLE_H / 2
        ):
            self.ball_x = learner_x + PADDLE_W
            self.ball_dx *= -1
            reward += 1.0

            hit_pos = (self.ball_y - self.learner_y) / (PADDLE_H / 2)
            self.ball_dy += hit_pos * 2
            self.ball_dy = np.clip(self.ball_dy, -8, 8)

        # ---- OPPONENT PADDLE ----
        opponent_x = WIDTH - 40 - PADDLE_W

        if (
            self.ball_dx > 0
            and opponent_x <= self.ball_x <= opponent_x + PADDLE_W
            and abs(self.ball_y - self.opponent_y) <= PADDLE_H / 2
        ):
            self.ball_x = opponent_x - 1
            self.ball_dx *= -1

            hit_pos = (self.ball_y - self.opponent_y) / (PADDLE_H / 2)
            self.ball_dy += hit_pos * 2
            self.ball_dy = np.clip(self.ball_dy, -8, 8)

        # ---- GOL ----
        if self.ball_x < 0:
            reward -= 10.0
            terminated = True
            winner = "opponent"
        elif self.ball_x > WIDTH:
            reward += 10.0
            terminated = True
            winner = "learner"

        # ---- ZAMAN AŞIMI (yeni: sonsuz rally'yi engeller) ----
        if not terminated and self.steps >= MAX_EPISODE_STEPS:
            truncated = True

        # ---- REWARD SHAPING (küçültülmüş ağırlık) ----
        if USE_DISTANCE_SHAPING:
            distance = abs(self.ball_y - self.learner_y)
            if self.ball_dx < 0:
                if distance < self.previous_distance:
                    reward += DISTANCE_SHAPING_WEIGHT
                else:
                    reward -= DISTANCE_SHAPING_WEIGHT
            self.previous_distance = distance

        info = {}
        if winner is not None:
            info["winner"] = winner

        return self.get_learner_obs(), reward, terminated, truncated, info


# ============================================================
# İSTATİSTİK CALLBACK
# ============================================================

class SelfPlayStatsCallback(BaseCallback):
    """
    Her adımda reward/episode uzunluğunu biriktirir, episode bittiğinde
    (done=True) kazananı sayar. `get_and_reset_stats()` her checkpoint
    penceresi sonunda çağrılır.
    """

    def __init__(self):
        super().__init__()
        self.current_ep_reward = 0.0
        self.current_ep_len = 0

        self.episode_rewards = deque()
        self.episode_lengths = deque()
        self.learner_wins = 0
        self.opponent_wins = 0
        self.timeouts = 0

        # Self-play'in gerçekten PPO vs PPO olduğunu doğrulamak için
        # ilk N opponent action'ını örnekliyoruz.
        self._opponent_action_sample = []

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        self.current_ep_reward += float(rewards[0])
        self.current_ep_len += 1

        if dones[0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_len)

            winner = infos[0].get("winner")
            if winner == "learner":
                self.learner_wins += 1
            elif winner == "opponent":
                self.opponent_wins += 1
            else:
                self.timeouts += 1

            self.current_ep_reward = 0.0
            self.current_ep_len = 0

        return True

    def get_and_reset_stats(self):
        n_episodes = len(self.episode_rewards)
        stats = {
            "episodes": n_episodes,
            "learner_wins": self.learner_wins,
            "opponent_wins": self.opponent_wins,
            "timeouts": self.timeouts,
            "win_rate": (
                self.learner_wins / (self.learner_wins + self.opponent_wins)
                if (self.learner_wins + self.opponent_wins) > 0 else float("nan")
            ),
            "avg_reward": (
                sum(self.episode_rewards) / n_episodes if n_episodes else float("nan")
            ),
            "avg_length": (
                sum(self.episode_lengths) / n_episodes if n_episodes else float("nan")
            ),
        }

        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.learner_wins = 0
        self.opponent_wins = 0
        self.timeouts = 0

        return stats


# ============================================================
# YARDIMCI: opponent'ın GERÇEKTEN PPO olduğunu doğrula
# ============================================================

def verify_opponent_is_real_ppo(opponent, pong_env, n_samples=200):
    """
    Rastgele gözlemler üretip opponent.predict() çağırıyoruz ve
    action dağılımını yazdırıyoruz. Eğer bu bir "topu takip et" scripti
    olsaydı, action dağılımı observation'a çok katı/deterministik bir
    şekilde bağlı olurdu; gerçek bir PPO modeli çok daha zengin/öğrenilmiş
    bir politika sergiler ve bu fonksiyon opponent objesinin gerçekten
    bir stable_baselines3.PPO instance'ı olduğunu da doğrudan kontrol eder.
    """
    assert isinstance(opponent, PPO), (
        "HATA: opponent bir stable_baselines3.PPO instance'ı değil! "
        "Self-play scripted bir bot ile karışmış olabilir."
    )

    actions = []
    for _ in range(n_samples):
        obs = pong_env.observation_space.sample()
        act, _ = opponent.predict(obs, deterministic=True)
        actions.append(int(act))

    unique, counts = np.unique(actions, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}

    print("  [DOĞRULAMA] opponent tipi:", type(opponent))
    print(f"  [DOĞRULAMA] {n_samples} rastgele gözlem üzerinde action dağılımı: {dist}")
    print("  [DOĞRULAMA] -> opponent gerçek bir PPO modeli ve predict() ile karar veriyor.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(SAVE_DIR, exist_ok=True)

    print()
    print("==============================================")
    print("          PONG SELF-PLAY TRAINING v2 (FRESH)")
    print("==============================================")
    print()
    print("Mod: SIFIRDAN EĞİTİM (eski model kullanılmıyor)")
    print("Device: CPU (GPU kullanılmayacak)")
    print(f"Checkpoint aralığı: {CHECKPOINT_STEPS:,} timestep")
    print(f"Warmup (rakipsiz) süre: {WARMUP_STEPS:,} timestep")
    print(f"Max episode uzunluğu: {MAX_EPISODE_STEPS} step (timeout koruması)")
    print()

    env = DummyVecEnv([lambda: SelfPlayPong()])
    pong_env = env.envs[0]

    print("Yeni PPO modeli (learner) sıfırdan oluşturuluyor...")
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,          # keşfi teşvik et (sıfırdan öğrenirken önemli)
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
    )
    print("Learner hazır. Başlangıç timestep:", model.num_timesteps)

    # ---- WARMUP: ilk pencerede rakip yok (paddle sabit) ----
    print()
    print(f"Warmup başlıyor: ilk {WARMUP_STEPS:,} timestep rakipsiz "
          f"(opponent paddle sabit duruyor).")
    pong_env.set_opponent(None)
    opponent = None
    print()

    stats_callback = SelfPlayStatsCallback()

    try:
        while True:
            current_steps = model.num_timesteps
            in_warmup = current_steps < WARMUP_STEPS
            window_steps = (
                WARMUP_STEPS - current_steps if in_warmup else CHECKPOINT_STEPS
            )

            print("----------------------------------------------")
            print("Yeni pencere başlıyor." + ("  [WARMUP - rakipsiz]" if in_warmup else ""))
            print(f"Mevcut timestep: {current_steps:,}")
            print(f"Sonraki checkpoint: {current_steps + window_steps:,}")
            print("----------------------------------------------")
            print()

            t0 = time.time()
            model.learn(
                total_timesteps=window_steps,
                reset_num_timesteps=False,
                callback=stats_callback,
            )
            elapsed = time.time() - t0

            total = model.num_timesteps
            stats = stats_callback.get_and_reset_stats()

            checkpoint_path = os.path.join(SAVE_DIR, f"pong_selfplay_{total}")

            print()
            print("=== PENCERE İSTATİSTİKLERİ ===")
            print(f"  Toplam timestep         : {total:,}")
            print(f"  Süre                    : {elapsed:.1f} sn "
                  f"({CHECKPOINT_STEPS/elapsed:.0f} step/sn)")
            print(f"  Tamamlanan maç sayısı   : {stats['episodes']}")
            print(f"  Learner kazandı         : {stats['learner_wins']}")
            print(f"  Opponent kazandı        : {stats['opponent_wins']}")
            print(f"  Zaman aşımı (berabere)  : {stats['timeouts']}")
            print(f"  Learner win rate        : {stats['win_rate']:.2%}"
                  if stats['win_rate'] == stats['win_rate'] else
                  "  Learner win rate        : n/a (maç bitmedi)")
            print(f"  Ortalama reward/episode : {stats['avg_reward']:.3f}")
            print(f"  Ortalama episode uzunluk: {stats['avg_length']:.1f}")
            print("================================")

            print()
            print("Checkpoint kaydediliyor...")
            model.save(checkpoint_path)
            # play_vs_ai.py'nin kolayca en sonuncuyu bulması için:
            model.save(os.path.join(SAVE_DIR, "pong_selfplay_latest"))
            print(f"Kaydedildi: {checkpoint_path}.zip")
            print(f"(latest -> pong_selfplay_latest.zip olarak da güncellendi)")

            # ---- YENİ RAKİP: az önce eğitilenin DONDURULMUŞ kopyası ----
            # (Warmup'tan çıkarken de ilk gerçek rakip burada kuruluyor.)
            # Önce yeni opponent'ı yükleyip env'e ata, ESKİSİNİ SONRA sil.
            new_opponent = PPO.load(checkpoint_path + ".zip", device="cpu")
            pong_env.set_opponent(new_opponent)
            old_opponent = opponent
            opponent = new_opponent
            del old_opponent

            print()
            print("Yeni rakip (bir önceki learner snapshot'ı) hazır.")
            verify_opponent_is_real_ppo(opponent, pong_env)
            print("Self-play devam ediyor...")
            print()

    except KeyboardInterrupt:
        print()
        print()
        print("==============================================")
        print("          EĞİTİM MANUEL OLARAK DURDURULDU")
        print("==============================================")
        print()

        emergency_path = os.path.join(
            SAVE_DIR, f"pong_selfplay_INTERRUPTED_{model.num_timesteps}"
        )

        print("Son durum kaydediliyor...")
        model.save(emergency_path)
        model.save(os.path.join(SAVE_DIR, "pong_selfplay_latest"))

        print()
        print("Kaydedildi:", emergency_path + ".zip")
        print("(latest -> pong_selfplay_latest.zip olarak da güncellendi)")
        print()
        print("Bu dosyadan daha sonra devam edebilirsin.")

        env.close()
