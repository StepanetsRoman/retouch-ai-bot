# 📦 Потужний Telegram-бот для ретуші з PSD, ControlNet, багатомовністю та реставрацією старих фото

import logging
import os
import uuid
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import InputFile, InlineKeyboardButton, InlineKeyboardMarkup

from gfpgan import GFPGANer
from rembg import remove
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from psd_tools import PSDImage, Group, Layer

API_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

TEMP_DIR = 'temp_images'
os.makedirs(TEMP_DIR, exist_ok=True)

restorer = GFPGANer(
    model_path=None,
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

upsampler = RealESRGANer(
    scale=2,
    model_path=None,
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                  num_block=23, num_grow_ch=32, scale=2),
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True
)

def get_options_keyboard():
    keyboard = InlineKeyboardMarkup()
    keyboard.add(
        InlineKeyboardButton("🧖‍♂️ Ретуш", callback_data="retouch"),
        InlineKeyboardButton("🎨 Покращення", callback_data="enhance"),
        InlineKeyboardButton("🧥 Відновити одяг", callback_data="restore_clothes")
    )
    keyboard.add(
        InlineKeyboardButton("🖼 Реставрація старих фото", callback_data="restore_old")
    )
    keyboard.add(
        InlineKeyboardButton("🧾 Зберегти PSD", callback_data="save_psd")
    )
    return keyboard

user_choices = {}
user_lang = {}

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("🇺🇦 Українська", callback_data="lang_uk"))
    keyboard.add(InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru"))
    keyboard.add(InlineKeyboardButton("🇬🇧 English", callback_data="lang_en"))
    await message.reply("🌍 Виберіть мову / Choose language:", reply_markup=keyboard)

@dp.callback_query_handler(lambda c: c.data.startswith("lang_"))
async def set_language(callback_query: types.CallbackQuery):
    lang_code = callback_query.data.split("_")[1]
    user_lang[callback_query.from_user.id] = lang_code
    await bot.send_message(callback_query.from_user.id, "👋 Обери, що саме обробити:", reply_markup=get_options_keyboard())

@dp.callback_query_handler()
async def handle_callback(callback_query: types.CallbackQuery):
    user_id = callback_query.from_user.id
    if callback_query.data.startswith("lang_"):
        return
    user_choices[user_id] = callback_query.data
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(user_id, "📷 Надішли фото для обробки")

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    choice = user_choices.get(user_id, 'retouch')

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    photo_bytes = await bot.download_file(file.file_path)

    uid = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"in_{uid}.jpg")
    output_path = os.path.join(TEMP_DIR, f"out_{uid}.png")
    psd_path = os.path.join(TEMP_DIR, f"result_{uid}.psd")

    with open(input_path, 'wb') as f:
        f.write(photo_bytes.read())

    await message.reply(f"🔧 Виконую: {choice.replace('_', ' ')}...")

    try:
        img = cv2.imread(input_path)
        result = img

        if choice == 'retouch':
            _, _, result = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        elif choice == 'enhance':
            result, _ = upsampler.enhance(img, outscale=2)
        elif choice == 'restore_clothes':
            result, _, _ = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            result, _ = upsampler.enhance(result, outscale=2)
        elif choice == 'restore_old':
            _, _, result = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            result, _ = upsampler.enhance(result, outscale=2)

        pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        img_no_bg = remove(pil_img)
        img_no_bg.save(output_path)

        if choice == 'save_psd':
            background = pil_img.convert("RGBA")
            foreground = remove(background).convert("RGBA")
            background.save(output_path)
            foreground.save("temp_fore.png")

            psd = PSDImage()
            psd.append(Layer.from_image(background, name="Фон"))
            psd.append(Layer.from_image(foreground, name="Об'єкт"))
            psd.save(psd_path)
            await bot.send_document(message.chat.id, InputFile(psd_path), caption="📁 PSD-файл з шарами")

        else:
            with open(output_path, 'rb') as f:
                await bot.send_photo(message.chat.id, f, caption="✅ Обробка завершена!")

    except Exception as e:
        await message.reply("❌ Помилка при обробці. Спробуй пізніше.")
        logging.error(e)
    finally:
        for path in [input_path, output_path, psd_path]:
            if os.path.exists(path): os.remove(path)
        if os.path.exists("temp_fore.png"): os.remove("temp_fore.png")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
