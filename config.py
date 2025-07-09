# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_hard_to_guess_secret_key' # IMPORTANT: Change this in production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///site.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Flask-Mail configuration (for password reset)
    MAIL_SERVER = os.environ.get('MAIL_SERVER') or 'smtp.googlemail.com'
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS') is not None
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME') # Your email address
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD') # Your email password/app password
    ADMINS = ['your-email@example.com'] # Email to send admin notifications (e.g., new user signup)