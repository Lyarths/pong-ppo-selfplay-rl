# 🏓 Pong PPO Self-Play (Reinforcement Learning)

**Stable-Baselines3 (PPO)** ve **Gymnasium** kullanılarak sıfırdan eğitilen, kendi dondurulmuş geçmiş kopyalarına (*frozen snapshot*) karşı sonsuz **Self-Play** maçları yaparak ustalaşan yapay zeka projesi ve insanlara karşı oynanabilen Pygame arayüzü.

---

## ✨ Özellikler

- 🤖 **Gerçek Self-Play Eğitimi (`train_selfplay.py`):**
  - Öğrenen PPO modeli, kendi geçmiş snapshot kopyasına karşı maç yapar.
  - Her checkpoint adımında kazanma oranı (*win rate*), ortalama ödül (*reward*) ve rally uzunluğu raporlanır; ardından güncel model yeni rakip olarak atanır.
  - **Ödül Şekillendirme (Reward Shaping) & Stokastik Rakip:** Ajanın "ortada sabit durma" veya tek bir rakip stiline ezberleme (*overfit*) sorunlarını önleyen geliştirilmiş v2 eğitim döngüsü.
  - `Ctrl+C` ile durdurulduğunda modeli kaybetmeden güvenli şekilde `./checkpoints` klasörüne kaydeder.
- 🎮 **İnsana Karşı Maç (`play_vs_ai.py`):**
  - `1`: Self-Play ile eğittiğiniz PPO modeline karşı oynayın.
  - `2`: Topun duvar sekmelerini aynalama yöntemiyle önceden hesaplayan sert bota (*scripted AI*) karşı oynayın.
- 🐢 **Klasik Pong (`classic_pong_turtle.py`):** Python `turtle` modülü ile hazırlanmış iki kişilik klasik Pong.

---

## 🚀 Kurulum ve Kullanım

```bash
pip install stable-baselines3 gymnasium numpy pygame
```

### 1. Yapay Zekayı Self-Play ile Eğitme
```bash
python train_selfplay.py
```

### 2. Eğitilen Modele veya Sert Bota Karşı Oynama
```bash
python play_vs_ai.py
```
*(Kontroller: `W` / `S` veya `Yukarı Ok` / `Aşağı Ok`)*
