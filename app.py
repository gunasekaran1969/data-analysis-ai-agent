"""
Data Analysis AI Agent - Main Application
A comprehensive AI-powered data analysis platform with visualization and insights generation
"""

import os
import json
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from agent.analyzer import DataAnalyzer
from agent.insights import InsightsGenerator
from agent.visualizer import DataVisualizer

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json', 'txt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize AI components
analyzer = DataAnalyzer()
insights_gen = InsightsGenerator()
visualizer = DataVisualizer()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_data(filepath):
    """Load data from file"""
    try:
        ext = filepath.rsplit('.', 1)[1].lower()
        
        if ext == 'csv':
            df = pd.read_csv(filepath)
        elif ext == 'xlsx':
            df = pd.read_excel(filepath)
        elif ext == 'json':
            df = pd.read_json(filepath)
        else:
            return None, "Unsupported file format"
        
        return df, None
    except Exception as e:
        return None, str(e)


@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use: CSV, XLSX, JSON'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        # Load and validate data
        df, error = load_data(filepath)
        if error:
            os.remove(filepath)
            return jsonify({'error': f'Failed to load file: {error}'}), 400
        
        # Get basic info
        info = {
            'filename': file.filename,
            'filepath': filepath,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'data': info
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Analyze uploaded data"""
    try:
        data = request.json
        filepath = data.get('filepath')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        # Load data
        df, error = load_data(filepath)
        if error:
            return jsonify({'error': f'Failed to load file: {error}'}), 400
        
        # Run analysis
        analysis_results = analyzer.analyze(df)
        
        return jsonify({
            'success': True,
            'analysis': analysis_results
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/insights', methods=['POST'])
def generate_insights():
    """Generate AI insights from data"""
    try:
        data = request.json
        filepath = data.get('filepath')
        analysis = data.get('analysis', {})
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        # Load data
        df, error = load_data(filepath)
        if error:
            return jsonify({'error': f'Failed to load file: {error}'}), 400
        
        # Generate insights
        insights = insights_gen.generate(df, analysis)
        
        return jsonify({
            'success': True,
            'insights': insights
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Insights generation failed: {str(e)}'}), 500


@app.route('/api/visualize', methods=['POST'])
def create_visualization():
    """Create data visualizations"""
    try:
        data = request.json
        filepath = data.get('filepath')
        viz_type = data.get('type', 'overview')
        column = data.get('column')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        # Load data
        df, error = load_data(filepath)
        if error:
            return jsonify({'error': f'Failed to load file: {error}'}), 400
        
        # Create visualization
        viz_data = visualizer.create(df, viz_type, column)
        
        return jsonify({
            'success': True,
            'visualization': viz_data
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Visualization failed: {str(e)}'}), 500


@app.route('/api/statistics', methods=['POST'])
def get_statistics():
    """Get detailed statistics"""
    try:
        data = request.json
        filepath = data.get('filepath')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        # Load data
        df, error = load_data(filepath)
        if error:
            return jsonify({'error': f'Failed to load file: {error}'}), 400
        
        # Calculate statistics
        stats = {
            'describe': df.describe().to_dict(),
            'correlations': df.corr().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
            'skewness': df.skew().to_dict(),
            'kurtosis': df.kurtosis().to_dict()
        }
        
        return jsonify({
            'success': True,
            'statistics': stats
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Statistics failed: {str(e)}'}), 500


@app.route('/api/anomalies', methods=['POST'])
def detect_anomalies():
    """Detect anomalies in data"""
    try:
        data = request.json
        filepath = data.get('filepath')
        column = data.get('column')
        method = data.get('method', 'zscore')  # zscore, iqr, isolation_forest
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        # Load data
        df, error = load_data(filepath)
        if error:
            return jsonify({'error': f'Failed to load file: {error}'}), 400
        
        if column not in df.columns:
            return jsonify({'error': f'Column {column} not found'}), 400
        
        # Detect anomalies
        anomalies = analyzer.detect_anomalies(df, column, method)
        
        return jsonify({
            'success': True,
            'anomalies': anomalies
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Anomaly detection failed: {str(e)}'}), 500


@app.route('/api/export', methods=['POST'])
def export_report():
    """Export analysis report"""
    try:
        data = request.json
        filepath = data.get('filepath')
        format_type = data.get('format', 'json')  # json, csv
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        # Load data
        df, error = load_data(filepath)
        if error:
            return jsonify({'error': f'Failed to load file: {error}'}), 400
        
        # Generate report
        analysis = analyzer.analyze(df)
        insights = insights_gen.generate(df, analysis)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'source_file': os.path.basename(filepath),
            'analysis': analysis,
            'insights': insights
        }
        
        if format_type == 'json':
            export_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(export_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return jsonify({
            'success': True,
            'report': report,
            'export_file': export_file
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV', 'production') == 'development'
    )
