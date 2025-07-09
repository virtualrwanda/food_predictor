import pandas as pd
import joblib
import sklearn
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, current_app
import os
import numpy as np
from datetime import datetime
from flask_mail import Mail, Message
from flask_login import LoginManager, login_user, logout_user, current_user, login_required, UserMixin
from flask_caching import Cache
from functools import wraps
import json
import secrets
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from custom modules
from config import Config
from models import db, User, Prediction
from forms import LoginForm, SignupForm, ForgotPasswordForm, ResetPasswordForm

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Flask extensions
db.init_app(app)
mail = Mail(app)

# Initialize caching
app.config['CACHE_TYPE'] = 'SimpleCache'
cache = Cache(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Flask-Login User Loader
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))  # Updated for SQLAlchemy 2.0 compatibility

# Model Loading
MODEL_DIR = "trained_models"
MODEL_FILENAME = "Linear_Regression_pipeline.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

model_pipeline = None
try:
    logger.info(f"Current scikit-learn version: {sklearn.__version__}")
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Model '{MODEL_FILENAME}' loaded successfully.")
        # Test prediction to validate model
        test_data = pd.DataFrame([{
            'year': 2023, 'month': 6, 'day_of_week': 2, 'day_of_year': 180,
            'admin1': 'Kigali City', 'admin2': 'Kigali', 'market': 'Main Market',
            'category': 'Cereals', 'commodity': 'Maize', 'unit': 'KG', 'pricetype': 'Retail'
        }])
        try:
            prediction = model_pipeline.predict(test_data)[0]
            logger.info(f"Test prediction (USD price): {prediction}")
            logger.info(f"Test prediction (RWF price): {prediction * 1250}")
        except Exception as e:
            logger.error(f"Test prediction failed: {e}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Prediction functionality will be unavailable.")
except joblib.exceptions.JobLibVersionError as e:
    logger.error(f"Version mismatch error: {e}. Ensure scikit-learn version matches the one used to save the model (likely 1.2.2).")
except Exception as e:
    logger.error(f"An error occurred while loading the model: {e}")

# Define expected model features
EXPECTED_FEATURES = [
    'year', 'month', 'day_of_week', 'day_of_year',
    'admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'pricetype'
]

# Exchange Rate (Fixed for demonstration)
USD_TO_RWF_EXCHANGE_RATE = 1250

# Admin Required Decorator
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Data Loading for Visualization
DATASET_PATH = os.path.join('data', 'food_prices_data.csv')

@cache.memoize(timeout=3600)  # Cache for 1 hour
def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH, low_memory=False)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            if df.empty:
                flash("Error: All dates were invalid or missing after parsing.", 'danger')
                logger.warning("Dataset is empty after date parsing.")
                return pd.DataFrame()
        for col in ['price', 'usdprice', 'latitude', 'longitude', 'market_id', 'commodity_id']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        placeholder_values = [
            '#item+name', '#geo+lat', '#geo+lon', '#item+code', '#item+unit',
            '#item+price+flag', '#item+price+type', '#currency+code',
            '#adm1+name', '#adm2+name', '#loc+market+name', '#loc+market+code'
        ]
        for col in df.columns:
            df = df[~df[col].isin(placeholder_values)]
        df.dropna(subset=['price'], inplace=True)
        logger.info(f"Dataset '{DATASET_PATH}' loaded successfully for visualization.")
        return df
    except FileNotFoundError:
        logger.error(f"Dataset not found at {DATASET_PATH}.")
        flash(f"Error: Dataset not found at {DATASET_PATH}.", 'danger')
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        flash(f"Error loading dataset: {str(e)}", 'danger')
        return pd.DataFrame()

# Plot Generation Function
def create_dynamic_price_trend_plot(df, commodity, admin1, pricetype, market=None, start_date=None, end_date=None):
    if df.empty:
        return {}
    filtered_df = df.copy()
    if commodity:
        filtered_df = filtered_df[filtered_df['commodity'] == commodity]
    if admin1:
        filtered_df = filtered_df[filtered_df['admin1'] == admin1]
    if pricetype:
        filtered_df = filtered_df[filtered_df['pricetype'] == pricetype]
    if market:
        filtered_df = filtered_df[filtered_df['market'] == market]
    if start_date:
        filtered_df = filtered_df[filtered_df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df['date'] <= pd.to_datetime(end_date)]
    filtered_df = filtered_df.sort_values('date')
    if filtered_df.empty:
        return {}
    title = f"Price Trend of {commodity or 'All Commodities'} in {admin1 or 'All Regions'} ({pricetype or 'All Price Types'})"
    if market:
        title += f" at {market}"

    labels = filtered_df['date'].dt.strftime('%Y-%m-%d').tolist()
    prices = filtered_df['price'].tolist()

    chart_config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Price (RWF)",
                "data": prices,
                "borderColor": "#4e79a7",
                "backgroundColor": "#4e79a766",
                "fill": False,
                "tension": 0.1
            }]
        },
        "options": {
            "scales": {
                "x": {
                    "title": {
                        "display": True,
                        "text": "Date"
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": "Price (RWF)"
                    },
                    "beginAtZero": True
                }
            },
            "plugins": {
                "legend": {
                    "display": True
                },
                "title": {
                    "display": True,
                    "text": title
                }
            }
        }
    }
    return json.dumps(chart_config)

# Flask Routes
@app.route('/')
def index():
    df = load_dataset()
    commodities = sorted(df['commodity'].unique().tolist()) if not df.empty else []
    markets = sorted(df['market'].unique().tolist()) if not df.empty else []
    return render_template('index.html', current_user=current_user, commodities=commodities, markets=markets)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model_pipeline is None:
        return jsonify({"error": "Prediction model not loaded. Please contact support."}), 500

    try:
        json_data = request.get_json(force=True)
        input_df = pd.DataFrame([json_data])

        for feature in EXPECTED_FEATURES:
            if feature not in input_df.columns:
                input_df[feature] = 0 if feature in ['year', 'month', 'day_of_week', 'day_of_year'] else 'unknown'

        input_df = input_df[EXPECTED_FEATURES]
        predicted_usd_price = model_pipeline.predict(input_df)[0]
        predicted_rwf_price = predicted_usd_price * USD_TO_RWF_EXCHANGE_RATE

        # Save prediction to database
        prediction = Prediction(
            user_id=current_user.id,
            input_data=json_data,
            predicted_usd_price=predicted_usd_price,
            predicted_rwf_price=predicted_rwf_price
        )
        db.session.add(prediction)
        db.session.commit()
        logger.info(f"Prediction saved for user {current_user.username}: USD {predicted_usd_price}, RWF {predicted_rwf_price}")

        return jsonify({
            "predicted_usdprice": float(predicted_usd_price),
            "predicted_rwfprice": float(predicted_rwf_price),
            "input_data": json_data
        })

    except KeyError as e:
        return jsonify({"error": f"Missing required input feature: {e}. Expected all of: {EXPECTED_FEATURES}"}), 400
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return jsonify({"error": f"An unexpected error occurred during prediction: {str(e)}"}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            flash('Logged in successfully!', 'success')
            logger.info(f"User {user.username} logged in successfully.")
            return redirect(next_page or url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
            logger.warning("Failed login attempt.")
    return render_template('login.html', title='Login', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = SignupForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        logger.info(f"New user created: {user.username}")
        return redirect(url_for('login'))
    return render_template('signup.html', title='Sign Up', form=form)

@app.route('/logout')
@login_required
def logout():
    username = current_user.username
    logout_user()
    flash('You have been logged out.', 'info')
    logger.info(f"User {username} logged out.")
    return redirect(url_for('index'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password_request():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = ForgotPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_reset_email(user)
            flash('An email has been sent with instructions to reset your password.', 'info')
            logger.info(f"Password reset requested for {user.email}")
        else:
            flash('If an account with that email exists, a password reset email has been sent.', 'info')
            logger.info(f"Password reset requested for non-existent email: {form.email.data}")
        return redirect(url_for('login'))
    return render_template('forgot_password.html', title='Forgot Password', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        logger.warning("Invalid or expired password reset token.")
        return redirect(url_for('forgot_password_request'))

    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        logger.info(f"Password reset for user {user.username}")
        return redirect(url_for('login'))
    return render_template('reset_password.html', title='Reset Password', form=form)

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender=current_app.config['MAIL_USERNAME'],
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    try:
        mail.send(msg)
        logger.info(f"Password reset email sent to {user.email}")
    except Exception as e:
        logger.error(f"Failed to send email to {user.email}: {e}")
        flash('Failed to send password reset email. Please check server configuration or try again later.', 'danger')

@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    users = User.query.all()
    return render_template('admin_dashboard.html', title='Admin Dashboard', users=users, current_user=current_user)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    user_to_delete = db.session.get(User, user_id)
    if not user_to_delete:
        flash('User not found.', 'danger')
        logger.warning(f"Attempt to delete non-existent user ID {user_id}.")
        return redirect(url_for('admin_dashboard'))
    if user_to_delete.is_admin:
        flash('Cannot delete an admin user directly from this interface.', 'danger')
        logger.warning(f"Attempt to delete admin user {user_to_delete.username} blocked.")
        return redirect(url_for('admin_dashboard'))
    if user_to_delete.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        logger.warning(f"User {current_user.username} attempted to delete their own account.")
        return redirect(url_for('admin_dashboard'))
    db.session.delete(user_to_delete)
    db.session.commit()
    flash(f'User "{user_to_delete.username}" has been deleted.', 'success')
    logger.info(f"User {user_to_delete.username} deleted by {current_user.username}")
    return redirect(url_for('admin_dashboard'))

@app.route('/visualize', methods=['GET', 'POST'])
@login_required
def visualize_data():
    df = load_dataset()
    plot_json = {}

    commodities = sorted(df['commodity'].unique().tolist()) if not df.empty else []
    admin1s = sorted(df['admin1'].unique().tolist()) if not df.empty else []
    pricetypes = sorted(df['pricetype'].unique().tolist()) if not df.empty else []
    markets = sorted(df['market'].unique().tolist()) if not df.empty else []

    selected_commodity = 'Maize'
    selected_admin1 = 'Kigali City'
    selected_pricetype = 'Retail'
    selected_market = None
    selected_start_date = None
    selected_end_date = None

    if request.method == 'POST':
        selected_commodity = request.form.get('commodity', selected_commodity)
        selected_admin1 = request.form.get('admin1', selected_admin1)
        selected_pricetype = request.form.get('pricetype', selected_pricetype)
        selected_market = request.form.get('market') or None
        selected_start_date = request.form.get('start_date') or None
        selected_end_date = request.form.get('end_date') or None

    if df.empty:
        flash('Could not load data for visualization or dataset is empty.', 'danger')
        logger.warning("Visualization failed due to empty dataset.")
    else:
        plot_json = create_dynamic_price_trend_plot(
            df,
            selected_commodity,
            selected_admin1,
            selected_pricetype,
            selected_market,
            selected_start_date,
            selected_end_date
        )
        if not plot_json:
            flash('No data available for the selected filters. Try different options.', 'warning')
            logger.warning("No data available for selected visualization filters.")

    return render_template('visualization.html',
                           plot_json=plot_json,
                           current_user=current_user,
                           commodities=commodities,
                           admin1s=admin1s,
                           pricetypes=pricetypes,
                           markets=markets,
                           selected_commodity=selected_commodity,
                           selected_admin1=selected_admin1,
                           selected_pricetype=selected_pricetype,
                           selected_market=selected_market,
                           selected_start_date=selected_start_date,
                           selected_end_date=selected_end_date)

@app.route('/prediction_history', methods=['GET'])
@login_required
def prediction_history():
    # Get filter parameters
    selected_commodity = request.args.get('commodity')
    selected_start_date = request.args.get('start_date')
    selected_end_date = request.args.get('end_date')

    # Query all predictions for the current user
    query = Prediction.query.filter_by(user_id=current_user.id)

    # Apply filters
    if selected_commodity:
        query = query.filter(Prediction.input_data['commodity'].astext == selected_commodity)
    if selected_start_date:
        query = query.filter(Prediction.timestamp >= pd.to_datetime(selected_start_date))
    if selected_end_date:
        query = query.filter(Prediction.timestamp <= pd.to_datetime(selected_end_date))

    predictions = query.order_by(Prediction.timestamp.desc()).all()

    # Get unique commodities from predictions for filter dropdown
    commodities = sorted({pred.input_data.get('commodity', '') for pred in Prediction.query.filter_by(user_id=current_user.id).all() if pred.input_data.get('commodity')})

    if not predictions:
        flash('No predictions found for the selected filters.', 'info')

    # Prepare data for visualization
    labels = [pred.timestamp.strftime('%Y-%m-%d %H:%M:%S') for pred in predictions]
    rwf_prices = [pred.predicted_rwf_price for pred in predictions]
    usd_prices = [pred.predicted_usd_price for pred in predictions]

    # Create Chart.js configuration for dual-axis line chart
    chart_config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Price (RWF)",
                    "data": rwf_prices,
                    "borderColor": "#4e79a7",
                    "backgroundColor": "#4e79a766",
                    "fill": False,
                    "yAxisID": "y-rwf",
                    "tension": 0.1
                },
                {
                    "label": "Price (USD)",
                    "data": usd_prices,
                    "borderColor": "#f28e38",
                    "backgroundColor": "#f28e3866",
                    "fill": False,
                    "yAxisID": "y-usd",
                    "tension": 0.1
                }
            ]
        },
        "options": {
            "scales": {
                "x": {
                    "title": {
                        "display": True,
                        "text": "Timestamp"
                    }
                },
                "y-rwf": {
                    "type": "linear",
                    "position": "left",
                    "title": {
                        "display": True,
                        "text": "Price (RWF)"
                    },
                    "beginAtZero": True
                },
                "y-usd": {
                    "type": "linear",
                    "position": "right",
                    "title": {
                        "display": True,
                        "text": "Price (USD)"
                    },
                    "beginAtZero": True,
                    "grid": {
                        "drawOnChartArea": False
                    }
                }
            },
            "plugins": {
                "legend": {
                    "display": True
                },
                "title": {
                    "display": True,
                    "text": f"Prediction History for {current_user.username}"
                }
            }
        }
    }

    plot_json = json.dumps(chart_config)
    return render_template('prediction_history.html',
                          plot_json=plot_json,
                          predictions=predictions,
                          current_user=current_user,
                          commodities=commodities,
                          selected_commodity=selected_commodity,
                          selected_start_date=selected_start_date,
                          selected_end_date=selected_end_date)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Creates User and Prediction tables
        if not User.query.filter_by(username='admin').first():
            admin_password = secrets.token_urlsafe(16)
            admin_user = User(username='admin', email='admin@example.com', is_admin=True)
            admin_user.set_password(admin_password)
            db.session.add(admin_user)
            db.session.commit()
            logger.info("\n--- Initial Setup ---")
            logger.info("Default admin user created:")
            logger.info("Username: admin")
            logger.info(f"Password: {admin_password}")
            logger.info("!!! PLEASE CHANGE THIS PASSWORD IMMEDIATELY AFTER FIRST LOGIN IN PRODUCTION !!!")
            logger.info("---------------------\n")
        if not User.query.first():
            default_user = User(username='testuser', email='test@example.com', is_admin=False)
            default_user.set_password('testpass')
            db.session.add(default_user)
            db.session.commit()
            logger.info("Default regular user created: username 'testuser', password 'testpass'")
    app.run(debug=os.getenv('FLASK_DEBUG', 'True') == 'True', host='0.0.0.0', port=5000)