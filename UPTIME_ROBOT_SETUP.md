# 🔄 UptimeRobot Setup - Держим бекенд активным

## 📌 Зачем это нужно?

На **бесплатном плане Render**:

- Если нет запросов в течение **15 минут** → сервис засыпает
- Следующий запрос будет медленным (холодный старт 30-60сек)

**UptimeRobot** решает это:

- ✅ Каждые 5 минут пингует ваш сервис
- ✅ Бекенд остается активным 24/7
- ✅ Бесплатный мониторинг + алерты при падении

---

## 🚀 ШАГИ ПОДКЛЮЧЕНИЯ

### Шаг 1: Создать аккаунт на UptimeRobot

1. Перейти: https://uptimerobot.com
2. Нажать **"Sign Up"** (бесплатно)
3. Подтвердить email

### Шаг 2: Добавить монитор для вашего приложения

1. Нажать **"Add New Monitor"**
2. Выбрать тип: **"HTTP(s)"**
3. Заполнить форму:

```
Monitor Name:              SafetyVision Health Check
Monitor Type:              HTTP(s)
URL (or IP):              https://safetyvision-guard.onrender.com/health
Monitoring Interval:       5 minutes (или 10)
HTTP Method:              GET
```

### Шаг 3: Дополнительные настройки (опционально)

**Advanced settings:**

- ✅ Enable keyword monitoring
  - Keyword to look for: `healthy`
  - (сервис вернет `"status":"healthy"` если всё OK)

**Notifications:**

- ✅ Отметить где получать алерты (Email, Telegram, Slack)
- ✅ При падении - сразу уведомит

### Шаг 4: Нажать **"Create Monitor"**

---

## 📊 ВАШ ЗДОРОВЫЙ ENDPOINT

Ваше приложение уже имеет `/health`:

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "mode": config.mode.value,
        "models": {...},
        "telegram": telegram.enabled,
        "version": "2.1.0",
    }
```

**Что возвращает:**

```json
{
  "status": "healthy",
  "mode": "demo",
  "telegram": true,
  "version": "2.1.0"
}
```

---

## 🔧 РЕКОМЕНДУЕМЫЕ НАСТРОЙКИ

| Параметр         | Значение  | Причина                                |
| ---------------- | --------- | -------------------------------------- |
| **Интервал**     | 5 минут   | Гарантирует активность, не перегружает |
| **Timeout**      | 30 сек    | Достаточно для холодного старта        |
| **Keyword**      | `healthy` | Подтверждает работоспособность         |
| **Notification** | На email  | Узнаете о проблемах сразу              |

---

## 🎯 РАСЧЕТ ЭКОНОМИИ

**Бесплатный план UptimeRobot:**

- ✅ 1 монитор (достаточно!)
- ✅ Интервал 5 минут
- ✅ Хранение данных 7 дней
- ✅ Email уведомления

**Что это даст:**

- ~288 пингов в день (~2.4 MB трафика)
- Бекенд ВСЕГДА активен
- Не нужно платить за Render Premium

---

## 🧪 ТЕСТИРОВАНИЕ

### 1. После создания монитора, проверить статус:

```bash
curl https://safetyvision-guard.onrender.com/health
```

Результат:

```json
{
  "status": "healthy",
  "mode": "demo",
  "models": {
    "people": "loaded",
    "ppe": "loaded",
    "fire": "loaded",
    "spill": "loaded",
    "fall": "loaded"
  },
  "telegram": true,
  "version": "2.1.0"
}
```

### 2. В UptimeRobot Dashboard:

Должен показать:

- ✅ Status: **Up**
- 📊 Uptime: **100%**
- 🟢 Last check: **Just now**

---

## 🔐 ДОПОЛНИТЕЛЬНЫЕ ВАРИАНТЫ

### Вариант A: Telegram уведомления (бесплатно)

1. В UptimeRobot → Settings → Notification Channels
2. **Add** → Telegram
3. Следуй инструкциям для привязки бота

### Вариант B: Slack уведомления

1. UptimeRobot → Settings → Notification Channels
2. **Add** → Slack
3. Авторизуй приложение

### Вариант C: Webhook (для логирования)

Если нужны кастомные действия при падении:

```
Webhook URL: https://ваш-сервис/api/uptime-ping
Method: POST
```

---

## 💡 PRO TIPS

### Если нужна большая надежность:

1. **Разные мониторы:**
   - `/health` - базовая проверка
   - `/api/detect-image` - тестовая загрузка изображения
   - `/ws` - проверка WebSocket

2. **Статус пейж:**
   - UptimeRobot создает публичную страницу: `status.uptimerobot.com`
   - Поделитесь со командой/клиентами

3. **Интеграция с CI/CD:**
   - Если монитор упал → автоматический рестарт на Render
   - Использовать Render API + UptimeRobot Webhook

---

## 🚨 ЕСЛИ ЧТО-ТО НЕ РАБОТАЕТ

### ❌ "Unreachable" в UptimeRobot

```bash
# Проверить доступ
curl -v https://safetyvision-guard.onrender.com/health

# Если CORS ошибка - уже решено в app.py (allow_origins=["*"])

# Если timeout - может быть холодный старт:
# Нужно 30-60 сек для первого запуска
```

### ❌ На Render статус "sleeping"

- Значит интервал UptimeRobot слишком большой
- Установите **5 минут** вместо 10

### ❌ "Keyword not found"

- Убедитесь что добавили `"healthy"` в поиск
- Проверьте что `/health` возвращает корректный JSON

---

## 📈 МОНИТОРИНГ ПАРАЛЛЕЛЬНО

Можно использовать несколько сервисов:

| Сервис                   | Бесплатно       | Функции                    |
| ------------------------ | --------------- | -------------------------- |
| **UptimeRobot**          | ✅ 1 монитор    | Пинг + уведомления         |
| **Statuspage.io**        | ✅ Базовая      | Публичная страница статуса |
| **Better Uptime**        | ✅ 10 мониторов | Более гибкий               |
| **Render Health Checks** | ✅ Встроено     | Внутренние проверки        |

---

## ✅ READY TO GO!

После настройки:

1. Откройте **UptimeRobot Dashboard**
2. Проверьте что монитор показывает **"Up"** 🟢
3. Можете забыть о проблеме "спящего сервиса"

**Результат:** Ваш SafetyVision AI будет доступен 24/7 без перерывов! 🚀
