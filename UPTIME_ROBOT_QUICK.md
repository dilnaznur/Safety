# ⚡ Быстрая настройка UptimeRobot (5 минут)

## ШАГ 1: Регистрация (1 мин)

Перейти: https://uptimerobot.com → **Sign Up** → подтвердить email

---

## ШАГ 2: Создать монитор (2 мин)

1. **Add New Monitor**
2. Заполнить:
   ```
   Monitor Name:     SafetyVision Backend
   Monitor Type:     HTTP(s)
   URL:              https://safetyvision-guard.onrender.com/ping
   Interval:         5 minutes
   Method:           GET
   ```
3. **Create Monitor**

---

## ШАГ 3: Проверить статус (1 мин)

- Должен показать 🟢 **Up**
- Если красный - проверьте URL в браузере:
  ```
  https://safetyvision-guard.onrender.com/health
  ```

---

## ШАГ 4: Уведомления (1 мин)

1. Settings → Notification Channels
2. **Add** → Email / Telegram / Slack
3. Подтвердить

---

## 📊 Доступные endpoints для мониторинга

| Endpoint    | Использование           | Скорость |
| ----------- | ----------------------- | -------- |
| **/ping**   | ⭐ UptimeRobot (легкий) | ~50ms    |
| **/health** | Полная информация       | ~100ms   |
| **/uptime** | Альтернатива            | ~50ms    |

**ИСПОЛЬЗУЙТЕ: `/ping`** - самый легкий и быстрый!

---

## ✅ Готово!

Ваше приложение будет:

- ✅ Пингован каждые 5 минут
- ✅ Остается активным 24/7
- ✅ Получите алерт при падении

**Результат:** Никогда не будет "холодного старта" 🚀
