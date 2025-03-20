#pip install python-telegram-bot
import logging
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# 📌 TOKEN DEL BOT (reemplázalo con el tuyo)
TELEGRAM_BOT_TOKEN = "token"

# 📌 URL de nuestra API en FastAPI (asegúrate de que FastAPI esté corriendo en este puerto)
API_URL = "http://127.0.0.1:8000/ask"

# 📌 Configurar logs
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# 📌 Función para manejar comandos como /start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hello! I am your E-commerce FAQ bot. Ask me anything!")

# 📌 Función para procesar mensajes y responder
async def handle_message(update: Update, context: CallbackContext) -> None:
    user_question = update.message.text  # Obtener la pregunta del usuario
    
    
    # 📌 Enviar la pregunta a la API
    response = requests.post(API_URL, json={"question": user_question})
    
    if response.status_code == 200:
        answer = response.text  # Respuesta en texto plano cambiar 
    else:
        answer = "Sorry, I couldn't get a response. Please try again later."

    # 📌 Enviar respuesta de vuelta al usuario
    await update.message.reply_text(answer)

# 📌 Función para iniciar el bot
def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # 📌 Comandos y mensajes
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 📌 Iniciar el bot
    print("🚀 Bot is running on Telegram...")
    app.run_polling()

if __name__ == "__main__":
    main()
