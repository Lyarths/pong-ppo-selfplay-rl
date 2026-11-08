"""
İnsana karşı Pong.

pip install pygame stable-baselines3

Başlarken:
  1 = RL modeli (self-play ile eğitilen PPO, pong_selfplay_latest.zip)
  2 = Sert bot (topun sekmelerini önceden hesaplayan, acımasız scripted AI)

Kontroller: yukarı ok / aşağı ok (veya W / S)
"""

import os
import sys
import numpy as np
import pygame

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
MODEL_PATH = os.path.join(SAVE_DIR, "pong_selfplay_latest.zip")

WIDTH = 800
HEIGHT = 500
PADDLE_W = 15
PADDLE_H = 80
PADDLE_SPEED = 7
BALL_SPEED = 6

FPS = 60


def get_ai_obs(ball_x, ball_y, ball_dx, ball_dy, ai_y, human_y):
    return np.array([
        ball_x / (WIDTH / 2) - 1,
        ball_y / (HEIGHT / 2) - 1,
        ball_dx / BALL_SPEED,
        ball_dy / BALL_SPEED,
        ai_y / (HEIGHT / 2) - 1,
        human_y / (HEIGHT / 2) - 1,
    ], dtype=np.float32)


def predict_ball_y_at_x(ball_x, ball_y, ball_dx, ball_dy, target_x):
    """
    Topun ileride target_x'e geldiğinde y konumunu, üst/alt duvar
    sekmelerini de hesaba katarak (aynalama yöntemiyle) hesaplar.
    Bu, 'sert bot'un neredeyse hiç kaçırmamasını sağlayan kısım.
    """
    if ball_dx == 0:
        return ball_y

    steps = (target_x - ball_x) / ball_dx
    if steps <= 0:
        return ball_y

    y = ball_y + ball_dy * steps

    # Duvarlardan sekmeleri aynalama (reflection) ile simüle et.
    period = 2 * HEIGHT
    y_mod = y % period
    if y_mod < 0:
        y_mod += period
    if y_mod > HEIGHT:
        y_mod = period - y_mod

    return y_mod


def hard_bot_action(ball_x, ball_y, ball_dx, ball_dy, bot_y, bot_x):
    """
    Acımasız scripted bot: top kendisine doğru geliyorsa nereye
    düşeceğini önceden hesaplar ve tam oraya gider. Top uzaklaşıyorsa
    ortaya doğru toparlanır. Hız sınırı öğrenen modelle aynı
    (PADDLE_SPEED) ama hedefi hiç şaşırmaz.
    """
    approaching = (bot_x < WIDTH / 2 and ball_dx < 0) or (bot_x > WIDTH / 2 and ball_dx > 0)

    if approaching:
        target_y = predict_ball_y_at_x(ball_x, ball_y, ball_dx, ball_dy, bot_x)
    else:
        target_y = HEIGHT / 2  # topu beklerken ortaya topla

    if bot_y < target_y - 4:
        return 2  # aşağı
    elif bot_y > target_y + 4:
        return 0  # yukarı
    return 1  # dur


def choose_mode():
    print()
    print("Rakip seç:")
    print("  1 = RL modeli (self-play ile eğitilen PPO)")
    print("  2 = Sert bot (acımasız, topu neredeyse hiç kaçırmaz)")
    choice = input("Seçim (1/2): ").strip()
    return "rl" if choice != "2" else "hard"


def main():
    mode = choose_mode()

    model = None
    if mode == "rl":
        if not os.path.exists(MODEL_PATH):
            print(f"Model bulunamadı: {MODEL_PATH}")
            print("Önce trainai.py ile en az bir checkpoint oluşturmalısın.")
            sys.exit(1)
        from stable_baselines3 import PPO
        print("Model yükleniyor:", MODEL_PATH)
        model = PPO.load(MODEL_PATH, device="cpu")
    else:
        print("Sert bot seçildi. İyi şanslar :)")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong - İnsan vs AI" + (" [HARD]" if mode == "hard" else ""))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 28)

    def reset():
        return {
            "ball_x": WIDTH / 2,
            "ball_y": HEIGHT / 2,
            "ball_dx": BALL_SPEED,
            "ball_dy": BALL_SPEED,
            "ai_y": HEIGHT / 2,
            "human_y": HEIGHT / 2,
        }

    s = reset()
    ai_score = 0
    human_score = 0

    ai_x = 40

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            s["human_y"] -= PADDLE_SPEED
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            s["human_y"] += PADDLE_SPEED
        s["human_y"] = float(np.clip(s["human_y"], PADDLE_H / 2, HEIGHT - PADDLE_H / 2))

        # AI hamlesi
        if mode == "rl":
            obs = get_ai_obs(s["ball_x"], s["ball_y"], s["ball_dx"], s["ball_dy"],
                              s["ai_y"], s["human_y"])
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = hard_bot_action(s["ball_x"], s["ball_y"], s["ball_dx"], s["ball_dy"],
                                      s["ai_y"], ai_x)

        if action == 0:
            s["ai_y"] -= PADDLE_SPEED
        elif action == 2:
            s["ai_y"] += PADDLE_SPEED
        s["ai_y"] = float(np.clip(s["ai_y"], PADDLE_H / 2, HEIGHT - PADDLE_H / 2))

        # Top
        s["ball_x"] += s["ball_dx"]
        s["ball_y"] += s["ball_dy"]

        if s["ball_y"] <= 0:
            s["ball_y"] = 0
            s["ball_dy"] *= -1
        if s["ball_y"] >= HEIGHT:
            s["ball_y"] = HEIGHT
            s["ball_dy"] *= -1

        # AI paddle (sol, x=40)
        if (
            s["ball_dx"] < 0
            and ai_x <= s["ball_x"] <= ai_x + PADDLE_W
            and abs(s["ball_y"] - s["ai_y"]) <= PADDLE_H / 2
        ):
            s["ball_x"] = ai_x + PADDLE_W
            s["ball_dx"] *= -1
            hit_pos = (s["ball_y"] - s["ai_y"]) / (PADDLE_H / 2)
            s["ball_dy"] += hit_pos * 2
            s["ball_dy"] = float(np.clip(s["ball_dy"], -8, 8))

        # İnsan paddle (sağ, x = WIDTH-40-PADDLE_W)
        human_x = WIDTH - 40 - PADDLE_W
        if (
            s["ball_dx"] > 0
            and human_x <= s["ball_x"] <= human_x + PADDLE_W
            and abs(s["ball_y"] - s["human_y"]) <= PADDLE_H / 2
        ):
            s["ball_x"] = human_x - 1
            s["ball_dx"] *= -1
            hit_pos = (s["ball_y"] - s["human_y"]) / (PADDLE_H / 2)
            s["ball_dy"] += hit_pos * 2
            s["ball_dy"] = float(np.clip(s["ball_dy"], -8, 8))

        # Gol
        if s["ball_x"] < 0:
            human_score += 1
            s = reset()
        elif s["ball_x"] > WIDTH:
            ai_score += 1
            s = reset()

        # Çizim
        screen.fill((10, 10, 10))
        pygame.draw.rect(screen, (255, 255, 255),
                          (ai_x, s["ai_y"] - PADDLE_H / 2, PADDLE_W, PADDLE_H))
        pygame.draw.rect(screen, (255, 255, 255),
                          (human_x, s["human_y"] - PADDLE_H / 2, PADDLE_W, PADDLE_H))
        pygame.draw.circle(screen, (255, 255, 255),
                            (int(s["ball_x"]), int(s["ball_y"])), 7)

        score_text = font.render(f"AI: {ai_score}   Sen: {human_score}", True, (255, 255, 255))
        screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
