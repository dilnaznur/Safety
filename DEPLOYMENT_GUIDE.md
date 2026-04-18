# Гайд по деплою на Vercel и Render

## 🚀 Деплой на Render

### 1. Подготовка репозитория

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/guard.git
git push -u origin main
```

### 2. Создание приложения на Render

1. Перейти на https://render.com
2. Нажать **"New" → "Web Service"**
3. Выбрать **"Connect a repository"** и авторизоваться через GitHub
4. Выбрать репозиторий `guard`
5. Заполнить параметры:
   - **Name**: `safetyvision-guard`
   - **Runtime**: Python 3.11
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port 8000`

### 3. Установка переменных окружения

В разделе **"Environment"** добавить:

- `BOT_TOKEN` - ваш Telegram токен
- `CHAT_ID` - ID вашего Telegram чата

### 4. Автоматический деплой

Render автоматически пересоберет приложение при push на `main`

---

## 🌐 Деплой на Vercel

### ⚠️ Важно

Vercel имеет **ограничения** для Python приложений:

- Максимальный размер функции: 50MB
- Timeout: 10 секунд для Free плана
- YOLO модели (~100MB) могут не поместиться

**Рекомендация**: Используйте Render для продакшена, Vercel - только для фронтенда.

### Если все же нужен Python на Vercel:

1. **Создать аккаунт**: https://vercel.com

2. **Установить Vercel CLI**:

```bash
npm install -g vercel
```

3. **Деплой**:

```bash
vercel
```

4. **Или через Git**: Подключить GitHub репозиторий в Vercel Dashboard

5. **Установить переменные**:
   - Перейти в Project Settings → Environment Variables
   - Добавить `BOT_TOKEN` и `CHAT_ID`

---

## 📊 Сравнение платформ

| Функция           | Render            | Vercel         |
| ----------------- | ----------------- | -------------- |
| Python приложения | ✅ Отлично        | ⚠️ Ограничено  |
| Docker            | ✅ Поддерживает   | ❌ Нет         |
| Размер моделей    | ✅ До 100GB       | ❌ До 50MB     |
| Бесплатный tier   | ✅ 0.5GB RAM      | ✅ 12x функции |
| Рекомендация      | **👍 Выбирайте!** | Для фронтенда  |

---

## 🔧 Для оптимизации на Render

Если приложение медленное, создайте `.renderignore`:

```
.git
.gitignore
.env.local
venv/
__pycache__/
*.pyc
.pytest_cache/
```

---

## ✅ Проверка после деплоя

```bash
# Для Render
curl https://safetyvision-guard.onrender.com/docs

# Для Vercel
curl https://YOUR_PROJECT.vercel.app/docs
```

Если видите Swagger UI документацию - все работает! ✅
