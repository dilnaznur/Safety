# Environment Variables Setup for Vercel & Render

## Required Variables

### Telegram Integration (Optional but recommended)

- `BOT_TOKEN`: Your Telegram Bot Token from @BotFather
  - Example: `8503867876:AAFzUvBVQJxehsM5auUkW6kUF4m6oym-kqY`
- `CHAT_ID`: Your Telegram Chat ID
  - Example: `1140942492`
  - How to get: Send `/start` to your bot, visit `https://api.telegram.org/bot<TOKEN>/getUpdates`, find `chat.id`

## Optional Variables

- `PORT`: Server port (automatically set by platform, default: 8000)
  - Render/Vercel automatically override this
- `PYTHONUNBUFFERED`: Set to `1` for real-time logs
  - `1`

## How to Set Variables

### On Render:

1. Go to your service page
2. Click "Settings" → "Environment"
3. Add each variable

### On Vercel:

1. Go to project settings
2. "Environment Variables" section
3. Paste all variables

## Deployment Checklist

- [ ] `.env` file has BOT_TOKEN and CHAT_ID locally (for testing)
- [ ] GitHub repository is pushed with latest code
- [ ] Render/Vercel environment variables are set
- [ ] `/health` endpoint returns 200 OK
- [ ] Telegram test works (if configured)

## Testing Telegram

Send a POST request to your deployed app:

```bash
curl -X POST https://your-app.onrender.com/api/telegram-test
```

Response should show:

```json
{
  "success": true
}
```
