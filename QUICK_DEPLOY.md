# 🚀 БЫСТРЫЙ СТАРТ: Деплой на Render и Vercel

## ШАГ 1️⃣: Подготовка GitHub репозитория

```powershell
# Если еще нет .git
git init
git add .
git commit -m "Initial commit for deployment"
git branch -M main

# Добавьте удаленный репозиторий (создайте на github.com/new)
git remote add origin https://github.com/YOUR_USERNAME/guard.git
git push -u origin main
```

---

## ШАГ 2️⃣: Деплой на RENDER (РЕКОМЕНДУЕМО ⭐)

### Вариант А: Через Dashboard

1. Перейти на **https://render.com** и авторизоваться через GitHub
2. Нажать **"New" → "Web Service"**
3. Выбрать репозиторий `guard`
4. Заполнить форму:

   ```
   Service Name: safetyvision-guard
   Branch: main
   Runtime: Python 3.11
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn app:app --host 0.0.0.0 --port 8000
   ```

5. **Advanced**: Отметить "Auto-deploy" если нужен CI/CD

### Переменные окружения на Render:

- `BOT_TOKEN` → скопируйте из `.env`
- `CHAT_ID` → скопируйте из `.env`
- `PORT` → `8000`

📍 **URL приложения**: `https://safetyvision-guard.onrender.com`

---

## ШАГ 3️⃣: Деплой на VERCEL (опционально)

### Для Python приложений (ограниченно):

```powershell
# Установить Vercel CLI
npm install -g vercel

# Деплой
vercel
```

Или через Vercel Dashboard:

1. https://vercel.com/new
2. Import Git Repository → выберите `guard`
3. Установите Environment Variables: `BOT_TOKEN`, `CHAT_ID`
4. Deploy

📍 **URL приложения**: `https://YOUR_PROJECT_NAME.vercel.app`

---

## ✅ ПРОВЕРКА ПОСЛЕ ДЕПЛОЯ

Откройте в браузере:

### Render

```
https://safetyvision-guard.onrender.com/docs
```

### Vercel

```
https://YOUR_PROJECT.vercel.app/docs
```

Должна открыться **Swagger UI документация** - это значит всё работает! ✨

---

## 🚨 РЕШЕНИЕ ПРОБЛЕМ

### ❌ "ModuleNotFoundError" при деплое

- Убедитесь, что `requirements.txt` в корне репозитория
- Все зависимости указаны в `requirements.txt`

### ❌ Приложение падает после деплоя

- Проверьте логи: Render Dashboard → Logs
- Убедитесь, что переменные окружения заданы

### ❌ Timeout на Vercel

- Перенесите приложение на **Render** (нет лимита на время)
- Vercel лучше для фронтенда, не для ML-приложений

### ❌ YOLO модели не загружаются

- Размер моделей может быть проблемой на Vercel
- На Render нет таких ограничений

---

## 📊 РЕКОМЕНДУЕМАЯ КОНФИГУРАЦИЯ

| Компонент              | Сервис          | Причина                                  |
| ---------------------- | --------------- | ---------------------------------------- |
| **Python API**         | Render          | Полная поддержка Python, нет ограничений |
| **Frontend (HTML/JS)** | Vercel          | Оптимален для статических файлов         |
| **Telegram Bot**       | Render          | Требует постоянное соединение            |
| **БД**                 | Render Postgres | Встроенная интеграция                    |

---

## 🎯 СЛЕДУЮЩИЕ ШАГИ

1. ✅ Создайте репозиторий на GitHub
2. ✅ Деплойте на Render
3. ✅ Проверьте `/docs` эндпоинт
4. ✅ Отправьте тестовое изображение
5. ✅ Проверьте Telegram оповещения

**Готово!** 🎉
