/**
 * SafetyVision AI — Internationalisation (i18n)
 * Supported: EN, RU, KZ
 * Usage:  i18n.setLang('ru');  i18n.t('key');
 */
const I18N = (() => {
    'use strict';

    const DICT = {
        // ─── HEADER / NAV ───────────────────────────────────
        "app.title":         { en: "SafetyVision AI",                ru: "SafetyVision AI",                     kz: "SafetyVision AI" },
        "app.subtitle":      { en: "Industrial Safety Monitoring",   ru: "Промышленный мониторинг безопасности", kz: "Өнеркәсіптік қауіпсіздік мониторингі" },
        "nav.dashboard":     { en: "Dashboard",                      ru: "Панель",                               kz: "Басқару тақтасы" },
        "nav.analytics":     { en: "Analytics",                      ru: "Аналитика",                            kz: "Аналитика" },
        "nav.settings":      { en: "Settings",                       ru: "Настройки",                            kz: "Баптаулар" },

        // ─── CONNECTION ─────────────────────────────────────
        "conn.online":       { en: "Online",                         ru: "Онлайн",                               kz: "Желіде" },
        "conn.offline":      { en: "Offline",                        ru: "Офлайн",                               kz: "Офлайн" },
        "conn.connected":    { en: "Connected to backend",           ru: "Подключено к серверу",                  kz: "Серверге қосылды" },
        "conn.reconnecting": { en: "Reconnecting...",                ru: "Переподключение...",                    kz: "Қайта қосылу..." },

        // ─── VIDEO PANEL ────────────────────────────────────
        "video.title":       { en: "Live Monitoring Feed",           ru: "Прямая трансляция",                    kz: "Тікелей мониторинг" },
        "video.upload":      { en: "Upload Video",                   ru: "Загрузить видео",                      kz: "Бейне жүктеу" },
        "video.webcam":      { en: "Webcam",                         ru: "Веб-камера",                           kz: "Веб-камера" },
        "video.image":       { en: "Upload Image",                   ru: "Загрузить фото",                       kz: "Сурет жүктеу" },
        "video.nosource":    { en: "No video source",                ru: "Нет источника видео",                  kz: "Бейне көзі жоқ" },
        "video.nosource_hint": { en: "Upload a video or image file, or connect your webcam to start real-time safety monitoring.",
                                 ru: "Загрузите видео, изображение или подключите веб-камеру для мониторинга.",
                                 kz: "Мониторинг бастау үшін бейне, сурет жүктеңіз немесе веб-камераны қосыңыз." },
        "video.live":        { en: "LIVE",                           ru: "ЖИВОЙ",                                kz: "ТІКЕЛЕЙ" },
        "video.video":       { en: "VIDEO",                          ru: "ВИДЕО",                                kz: "БЕЙНЕ" },
        "video.back_live":   { en: "Back to Live",                   ru: "Назад к трансляции",                   kz: "Тікелей эфирге оралу" },

        // ─── DETECTION MODES ────────────────────────────────
        "mode.title":        { en: "Detection Mode",                 ru: "Режим обнаружения",                    kz: "Анықтау режимі" },
        "mode.all":          { en: "All-in-One",                     ru: "Все сразу",                            kz: "Барлығы" },
        "mode.people":       { en: "People",                         ru: "Люди",                                 kz: "Адамдар" },
        "mode.ppe":          { en: "PPE",                            ru: "СИЗ",                                  kz: "ЖҚЖ" },
        "mode.fire":         { en: "Fire",                           ru: "Огонь",                                kz: "Өрт" },
        "mode.spill":        { en: "Spills",                         ru: "Разливы",                              kz: "Төгілулер" },
        "mode.fall":         { en: "Falls",                          ru: "Падения",                              kz: "Құлау" },

        // ─── STATS ──────────────────────────────────────────
        "stat.people":       { en: "Current People",                 ru: "Текущее кол-во людей",                 kz: "Қазіргі адам саны" },
        "stat.entered":      { en: "Entered",                        ru: "Вошли",                                kz: "Кірді" },
        "stat.exited":       { en: "Exited",                         ru: "Вышли",                                kz: "Шықты" },
        "stat.peak":         { en: "Peak",                           ru: "Макс.",                                kz: "Ең көп" },
        "stat.ppe":          { en: "PPE Compliance",                 ru: "Соответствие СИЗ",                     kz: "ЖҚЖ сәйкестігі" },
        "stat.fire":         { en: "Fire Risk Level",                ru: "Уровень пожарного риска",              kz: "Өрт қаупі деңгейі" },
        "stat.alerts":       { en: "Active Alerts",                  ru: "Активные оповещения",                  kz: "Белсенді ескертулер" },
        "stat.spills":       { en: "Spills Today",                   ru: "Разливы за сегодня",                   kz: "Бүгінгі төгілулер" },
        "stat.falls":        { en: "Falls Detected",                 ru: "Обнаружено падений",                   kz: "Құлау анықталды" },
        "stat.uptime":       { en: "Session Uptime",                 ru: "Время сессии",                         kz: "Сессия уақыты" },
        "stat.live":         { en: "Live Statistics",                ru: "Статистика в реальном времени",        kz: "Тікелей статистика" },

        // ─── FIRE RISK VALUES ───────────────────────────────
        "fire.safe":         { en: "Safe",                           ru: "Безопасно",                            kz: "Қауіпсіз" },
        "fire.high":         { en: "High",                           ru: "Высокий",                              kz: "Жоғары" },
        "fire.critical":     { en: "CRITICAL",                       ru: "КРИТИЧЕСКИЙ",                          kz: "ҚАУІПТІ" },

        // ─── ALERTS ─────────────────────────────────────────
        "alerts.title":      { en: "Recent Alerts",                  ru: "Последние оповещения",                 kz: "Соңғы ескертулер" },
        "alerts.clear":      { en: "Clear All",                      ru: "Очистить все",                         kz: "Барлығын тазалау" },
        "alerts.none":       { en: "No alerts yet. Start monitoring to see safety alerts here.",
                               ru: "Нет оповещений. Начните мониторинг для появления оповещений.",
                               kz: "Ескертулер жоқ. Мониторинг бастаңыз." },
        "alerts.dismiss":    { en: "Dismiss",                        ru: "Скрыть",                               kz: "Жабу" },

        // ─── QUICK ACTIONS ──────────────────────────────────
        "action.title":      { en: "Quick Actions",                  ru: "Быстрые действия",                    kz: "Жылдам әрекеттер" },
        "action.export":     { en: "Export Report",                  ru: "Экспорт отчёта",                      kz: "Есепті экспорттау" },
        "action.mute":       { en: "Mute Alerts",                    ru: "Без звука",                            kz: "Дыбыссыз" },
        "action.unmute":     { en: "Unmute Alerts",                  ru: "Включить звук",                       kz: "Дыбысты қосу" },
        "action.emergency":  { en: "Emergency Stop",                 ru: "Аварийная остановка",                  kz: "Шұғыл тоқтату" },
        "action.emergency_confirm": { en: "Are you sure you want to trigger EMERGENCY STOP?",
                                      ru: "Вы уверены, что хотите активировать АВАРИЙНУЮ ОСТАНОВКУ?",
                                      kz: "ШҰҒЫЛ ТОҚТАТУДЫ іске қосуды растайсыз ба?" },

        // ─── ANALYTICS ──────────────────────────────────────
        "analytics.alert_dist":  { en: "Alert Distribution",         ru: "Распределение оповещений",             kz: "Ескертулер таратылымы" },
        "analytics.timeline":    { en: "People Count Timeline",      ru: "Хронология присутствия людей",         kz: "Адам саны хронологиясы" },
        "analytics.ppe_chart":   { en: "PPE Compliance Over Time",   ru: "Соответствие СИЗ во времени",          kz: "ЖҚЖ сәйкестігі уақыт бойынша" },
        "analytics.summary":     { en: "Session Summary",            ru: "Итоги сессии",                         kz: "Сессия қорытындысы" },
        "analytics.total_alerts":{ en: "Total Alerts",               ru: "Всего оповещений",                     kz: "Барлық ескертулер" },
        "analytics.critical":    { en: "Critical Events",            ru: "Критические события",                  kz: "Қауіпті оқиғалар" },
        "analytics.tracked":     { en: "People Tracked",             ru: "Отслежено людей",                      kz: "Бақыланған адамдар" },
        "analytics.peak_occ":    { en: "Peak Occupancy",             ru: "Макс. загрузка",                       kz: "Ең көп адам" },
        "analytics.avg_ppe":     { en: "Avg PPE Compliance",         ru: "Средн. соотв. СИЗ",                    kz: "Орташа ЖҚЖ сәйкестігі" },
        "analytics.spill_events":{ en: "Spill Events",               ru: "Случаи разлива",                       kz: "Төгілу оқиғалары" },
        "analytics.fall_events": { en: "Fall Events",                ru: "Случаи падения",                       kz: "Құлау оқиғалары" },

        // ─── SETTINGS ──────────────────────────────────────
        "settings.connection":   { en: "Connection Settings",        ru: "Параметры подключения",                kz: "Қосылу баптаулары" },
        "settings.ws_url":       { en: "Backend WebSocket URL",      ru: "WebSocket URL сервера",                kz: "Сервер WebSocket URL" },
        "settings.reconnect":    { en: "Reconnect",                  ru: "Переподключиться",                     kz: "Қайта қосылу" },
        "settings.alert_title":  { en: "Alert Settings",             ru: "Настройки оповещений",                 kz: "Ескерту баптаулары" },
        "settings.sound":        { en: "Play sound on critical alerts",  ru: "Звук при критических оповещениях",  kz: "Қауіпті ескертулерде дыбыс" },
        "settings.notif":        { en: "Browser push notifications",     ru: "Push-уведомления браузера",         kz: "Браузер push-хабарламалары" },
        "settings.occupancy":    { en: "Max Occupancy Threshold",        ru: "Макс. порог присутствия",           kz: "Максималды адам шегі" },
        "settings.telegram":     { en: "Telegram Settings",              ru: "Настройки Telegram",                kz: "Telegram баптаулары" },
        "settings.tg_test":      { en: "Test Telegram",                  ru: "Тест Telegram",                     kz: "Telegram тексеру" },
        "settings.about":        { en: "About",                          ru: "О системе",                         kz: "Жүйе туралы" },
        "settings.about_text":   { en: "SafetyVision AI v2.0.0 — Industrial Safety Monitoring System",
                                   ru: "SafetyVision AI v2.0.0 — Система промышленного мониторинга безопасности",
                                   kz: "SafetyVision AI v2.0.0 — Өнеркәсіптік қауіпсіздік мониторинг жүйесі" },
        "settings.powered":      { en: "Powered by YOLOv8 · FastAPI · WebSocket",
                                   ru: "На базе YOLOv8 · FastAPI · WebSocket",
                                   kz: "YOLOv8 · FastAPI · WebSocket негізінде" },
        "settings.language":     { en: "Language",                       ru: "Язык",                               kz: "Тіл" },

        // ─── TOASTS / NOTIFICATIONS ─────────────────────────
        "toast.mode":        { en: "Detection mode:",                ru: "Режим обнаружения:",                   kz: "Анықтау режимі:" },
        "toast.upload_vid":  { en: "Uploading video...",             ru: "Загрузка видео...",                    kz: "Бейне жүктелуде..." },
        "toast.vid_ok":      { en: "Video loaded — streaming with detection", ru: "Видео загружено — трансляция с детекцией", kz: "Бейне жүктелді — анықтаумен трансляция" },
        "toast.vid_fail":    { en: "Failed to load video",           ru: "Ошибка загрузки видео",                kz: "Бейне жүктеу қатесі" },
        "toast.img_analyze": { en: "Analyzing image...",             ru: "Анализ изображения...",                kz: "Сурет талдануда..." },
        "toast.img_ok":      { en: "Image analyzed:",                ru: "Изображение проанализировано:",        kz: "Сурет талданды:" },
        "toast.img_fail":    { en: "Image analysis failed",          ru: "Ошибка анализа изображения",           kz: "Сурет талдау қатесі" },
        "toast.webcam_ok":   { en: "Webcam connected",               ru: "Веб-камера подключена",                kz: "Веб-камера қосылды" },
        "toast.webcam_fail": { en: "Webcam not available",           ru: "Веб-камера недоступна",                kz: "Веб-камера қолжетімсіз" },
        "toast.webcam_err":  { en: "Could not connect webcam",       ru: "Не удалось подключить веб-камеру",     kz: "Веб-камераны қосу мүмкін болмады" },
        "toast.export":      { en: "Report exported",                ru: "Отчёт экспортирован",                  kz: "Есеп экспортталды" },
        "toast.muted":       { en: "Alerts muted",                   ru: "Оповещения отключены",                 kz: "Ескертулер өшірілді" },
        "toast.unmuted":     { en: "Alerts unmuted",                 ru: "Оповещения включены",                  kz: "Ескертулер қосылды" },
        "toast.emergency":   { en: "EMERGENCY STOP ACTIVATED",       ru: "АВАРИЙНАЯ ОСТАНОВКА АКТИВИРОВАНА",     kz: "ШҰҒЫЛ ТОҚТАТУ ІСКЕ ҚОСЫЛДЫ" },
        "toast.live":        { en: "Switched to live mode",          ru: "Переключено на трансляцию",            kz: "Тікелей режимге ауысты" },
        "toast.video_play":  { en: "Playing uploaded video with detection", ru: "Воспроизведение видео с детекцией", kz: "Бейне анықтаумен ойнатылуда" },
        "toast.upload_mode": { en: "Upload analysis mode",           ru: "Режим анализа загрузки",               kz: "Жүктеу талдау режимі" },
        "toast.tg_ok":       { en: "Telegram test sent!",            ru: "Тест Telegram отправлен!",             kz: "Telegram тест жіберілді!" },
        "toast.tg_fail":     { en: "Telegram test failed",           ru: "Тест Telegram не удался",              kz: "Telegram тест сәтсіз" },
        "toast.upload_fail": { en: "Upload failed",                  ru: "Загрузка не удалась",                  kz: "Жүктеу сәтсіз" },

        // ─── DETECTION LABELS ───────────────────────────────
        "det.detections":    { en: "detections",                     ru: "обнаружений",                          kz: "анықтау" },
        "det.image_analysis":{ en: "Image analysis",                 ru: "Анализ изображения",                   kz: "Сурет талдау" },
    };

    let _lang = 'en';

    function t(key) {
        const entry = DICT[key];
        if (!entry) return key;
        return entry[_lang] || entry['en'] || key;
    }

    function setLang(lang) {
        lang = (lang || 'en').toLowerCase();
        if (!['en', 'ru', 'kz'].includes(lang)) lang = 'en';
        _lang = lang;
        try { localStorage.setItem('sv_lang', lang); } catch(e) {}
        applyAll();
    }

    function getLang() { return _lang; }

    function init() {
        try { _lang = localStorage.getItem('sv_lang') || 'en'; } catch(e) {}
        if (!['en', 'ru', 'kz'].includes(_lang)) _lang = 'en';
        applyAll();
    }

    /** Scan DOM for data-i18n attributes and update textContent. */
    function applyAll() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            const val = t(key);
            if (el.tagName === 'INPUT' && el.type !== 'checkbox') {
                el.placeholder = val;
            } else {
                el.textContent = val;
            }
        });
        // Also update data-i18n-title (tooltips)
        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            el.title = t(el.getAttribute('data-i18n-title'));
        });
    }

    return { t, setLang, getLang, init, applyAll, DICT };
})();
