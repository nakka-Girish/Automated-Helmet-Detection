from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename
import pandas as pd
import csv
from datetime import datetime
# Import functions from main.py
from main import process_image, process_video, process_live


app = Flask(__name__)
app.secret_key = 'girish@helmet-detection'


UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    process_type = request.form.get('type', '1')  # Default to image processing
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if process_type == '1':  # Image processing
        if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Process the image using the function from main.py
            try:
                import threading
                thread = threading.Thread(target=process_image, args=(file_path,))
                thread.daemon = True
                thread.start()
                return jsonify({'success': True, 'message': 'Image processing started', 'file': unique_filename})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    
    elif process_type == '2':  # Video processing
        if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Process the video using the function from main.py
            try:
                import threading
                thread = threading.Thread(target=process_video, args=(file_path,))
                thread.daemon = True
                thread.start()
                return jsonify({'success': True, 'message': 'Video processing started', 'file': unique_filename})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    
    return jsonify({'error': 'Invalid process type'}), 400


@app.route('/payment_login', methods=['GET', 'POST'])
def payment_login():
    if request.method == 'POST':
        mobile_number = request.form.get('mobile_number')
        # Check if mobile exists in CSV
        penalty_data = get_penalty_data(mobile_number)
        
        if penalty_data:
            session['mobile_number'] = mobile_number
            session['penalty_data'] = penalty_data
            return redirect(url_for('payment_gateway'))
        else:
            return render_template('payment_login.html', error="No penalty record found for this mobile number")
    
    return render_template('payment_login.html')

@app.route('/payment_gateway', methods=['GET'])
def payment_gateway():
    if 'mobile_number' not in session:
        return redirect(url_for('payment_login'))
    
    penalty_data = session.get('penalty_data')
    return render_template('payment_gateway.html', penalty_data=penalty_data)

@app.route('/process_payment', methods=['POST'])
def process_payment():
    if 'mobile_number' not in session:
        return redirect(url_for('payment_login'))
    
    payment_method = request.form.get('payment_method')
    amount_paid = float(request.form.get('amount'))
    mobile_number = session.get('mobile_number')
    penalty_data = session.get('penalty_data')
    
    # Update the CSV with payment information
    success, error = update_payment_record(mobile_number, amount_paid)
    
    if success:
 
        return render_template('payment_success.html', amount=amount_paid)
    else:
        return render_template('payment_gateway.html',penalty_data=penalty_data, error=f"Thisi is error: {error}")

def get_penalty_data(mobile_number):
    try:
        csv_path = os.path.join('data', 'database.csv')
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # Strip column names
        df['Mobile'] = df['Mobile'].astype(str).str.strip()  # Strip and ensure string type

        record = df[df['Mobile'] == mobile_number]
        
        if not record.empty:
            return {
                'name': str(record['Name'].values[0]),
                'vehicle_number': str(record['Registration'].values[0]),
                'penalty_amount': int(record['Challan'].values[0]),
                }
    except Exception as e:
        print(f"Error fetching penalty data: {e}")
    
    return None

def update_payment_record(mobile_number, amount_paid):
    """Update CSV file with payment information"""
    try:
        df = pd.read_csv('data/database.csv')
        df.columns = df.columns.str.strip()
        df['Mobile'] = df['Mobile'].astype(str).str.strip()  # Ensure type and strip whitespace
        mobile_number = str(mobile_number).strip()  # Ensure incoming number is also clean

        index = df[df['Mobile'] == mobile_number].index
        
        if len(index) > 0:
            current_challan = float(df.loc[index[0], 'Challan'])
            amount_paid = float(amount_paid)
            df.loc[index[0], 'Challan'] = current_challan - amount_paid
            df.to_csv('data/database.csv', index=False)
            return True, None  # Success
        else:
            return False, "Mobile number not found in CSV"

    except Exception as e:
        print(f"Error updating payment record: {e}")
        return False, str(e)





@app.route('/webcam-detection', methods=['POST'])
def webcam_detection():
    try:
        # Start webcam detection using the function from main.py
        import threading
        thread = threading.Thread(target=process_live, args=(True,))
        thread.daemon = True
        thread.start()
        return jsonify({'success': True, 'message': 'Webcam detection started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add route for cleanup (optional)
@app.route('/cleanup', methods=['POST'])
def cleanup():
    try:
       return jsonify({'success': True, 'message': 'Cleanup completed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='127.0.0.1', port=5000)

