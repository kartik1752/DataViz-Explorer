# app.py
from flask import Flask, render_template, request, send_file, jsonify, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import io
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/plots'
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store data (in production, use database or session storage)
df_global = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global df_global
    info = None
    preview = None
    stats_data = None
    numeric_cols = []
    categorical_cols = []
    plot_files = {}
    correlation_matrix = None
    
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            try:
                df_global = pd.read_csv(file)
                session['df_columns'] = list(df_global.columns)
                session['df_shape'] = df_global.shape
                
                # Preview first 5 rows
                preview = df_global.head().to_html(classes='table table-striped table-hover', index=False)
                
                # File Info
                info = {
                    'rows': df_global.shape[0],
                    'columns': df_global.shape[1],
                    'column_names': list(df_global.columns),
                    'missing_values': df_global.isnull().sum().to_dict(),
                    'file_size': f"{len(file.read()) / 1024:.2f} KB",
                    'data_types': {col: str(dtype) for col, dtype in df_global.dtypes.to_dict().items()}
                }
                file.seek(0)  # Reset file pointer after reading
                
                # Column types
                numeric_cols = df_global.select_dtypes(include='number').columns.tolist()
                categorical_cols = df_global.select_dtypes(exclude='number').columns.tolist()
                
                # Statistics
                numeric_stats = df_global[numeric_cols].describe().round(2) if numeric_cols else None
                categorical_stats = df_global[categorical_cols].describe() if categorical_cols else None
                
                # Additional statistics
                additional_stats = {}
                if numeric_cols:
                    for col in numeric_cols:
                        additional_stats[col] = {
                            'skewness': round(df_global[col].skew(), 2),
                            'kurtosis': round(df_global[col].kurtosis(), 2),
                            'variance': round(df_global[col].var(), 2)
                        }
                
                stats_data = {
                    'numeric': numeric_stats.to_html(classes='table table-bordered table-striped') if numeric_stats is not None else None,
                    'categorical': categorical_stats.to_html(classes='table table-bordered table-striped') if categorical_stats is not None else None,
                    'additional': additional_stats
                }
                
                # Generate visualizations
                plot_files = generate_visualizations(df_global, numeric_cols, categorical_cols)
                
                # Correlation matrix for numeric columns
                if len(numeric_cols) > 1:
                    plt.figure(figsize=(10, 8))
                    correlation = df_global[numeric_cols].corr()
                    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
                    plt.title('Correlation Matrix')
                    correlation_path = os.path.join(app.config['UPLOAD_FOLDER'], 'correlation.png')
                    plt.savefig(correlation_path)
                    plt.close()
                    plot_files['correlation'] = correlation_path
                    correlation_matrix = correlation.to_dict()
                
            except Exception as e:
                error_message = f"Error processing file: {str(e)}"
                return render_template('index.html', error=error_message)
    
    return render_template('index.html', info=info, preview=preview, stats=stats_data,
                           numeric_cols=numeric_cols, categorical_cols=categorical_cols, 
                           plot_files=plot_files, correlation_matrix=correlation_matrix)

@app.route('/filter_data', methods=['POST'])
def filter_data():
    global df_global
    if df_global is None:
        return jsonify({'error': 'No data available'})
    
    try:
        filters = request.get_json()
        filtered_df = df_global.copy()
        
        # Apply column filters
        for column, condition in filters.items():
            if column in filtered_df.columns:
                if condition.get('min') is not None:
                    filtered_df = filtered_df[filtered_df[column] >= condition['min']]
                if condition.get('max') is not None:
                    filtered_df = filtered_df[filtered_df[column] <= condition['max']]
                if condition.get('values') is not None:
                    filtered_df = filtered_df[filtered_df[column].isin(condition['values'])]
        
        # Get sample of filtered data
        sample_data = filtered_df.head(100).to_dict('records')
        
        return jsonify({
            'filtered_count': len(filtered_df),
            'sample_data': sample_data,
            'columns': list(filtered_df.columns)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    global df_global
    if df_global is None:
        return jsonify({'error': 'No data available'})
    
    try:
        plot_type = request.form.get('plot_type')
        x_axis = request.form.get('x_axis')
        y_axis = request.form.get('y_axis')
        hue = request.form.get('hue')
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'scatter' and x_axis and y_axis:
            if hue and hue in df_global.columns:
                sns.scatterplot(data=df_global, x=x_axis, y=y_axis, hue=hue)
            else:
                plt.scatter(df_global[x_axis], df_global[y_axis])
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.title(f'Scatter Plot: {x_axis} vs {y_axis}')
            
        elif plot_type == 'line' and x_axis and y_axis:
            if hue and hue in df_global.columns:
                for value in df_global[hue].unique():
                    subset = df_global[df_global[hue] == value]
                    plt.plot(subset[x_axis], subset[y_axis], label=value)
                plt.legend()
            else:
                plt.plot(df_global[x_axis], df_global[y_axis])
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.title(f'Line Plot: {x_axis} vs {y_axis}')
            
        elif plot_type == 'bar' and x_axis:
            if y_axis:
                if hue and hue in df_global.columns:
                    sns.barplot(data=df_global, x=x_axis, y=y_axis, hue=hue)
                else:
                    plt.bar(df_global[x_axis], df_global[y_axis])
                plt.ylabel(y_axis)
            else:
                value_counts = df_global[x_axis].value_counts()
                plt.bar(value_counts.index, value_counts.values)
                plt.ylabel('Count')
            plt.xlabel(x_axis)
            plt.title(f'Bar Chart: {x_axis}')
            plt.xticks(rotation=45)
            
        elif plot_type == 'histogram' and x_axis:
            plt.hist(df_global[x_axis].dropna(), bins=20, alpha=0.7)
            plt.xlabel(x_axis)
            plt.ylabel('Frequency')
            plt.title(f'Histogram: {x_axis}')
            
        elif plot_type == 'box' and x_axis:
            if y_axis:
                if hue and hue in df_global.columns:
                    sns.boxplot(data=df_global, x=x_axis, y=y_axis, hue=hue)
                else:
                    sns.boxplot(data=df_global, x=x_axis, y=y_axis)
                plt.ylabel(y_axis)
            else:
                sns.boxplot(data=df_global[x_axis].dropna())
                plt.ylabel(x_axis)
            plt.xlabel(x_axis if y_axis else '')
            plt.title(f'Box Plot: {x_axis}')
            plt.xticks(rotation=45)
            
        else:
            return jsonify({'error': 'Invalid plot parameters'})
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'custom_plot_{timestamp}.png'
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        return jsonify({'plot_url': f'/static/plots/{plot_filename}'})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_column_stats', methods=['POST'])
def get_column_stats():
    global df_global
    if df_global is None:
        return jsonify({'error': 'No data available'})
    
    column = request.form.get('column')
    if column not in df_global.columns:
        return jsonify({'error': 'Column not found'})
    
    try:
        if pd.api.types.is_numeric_dtype(df_global[column]):
            # Numerical column statistics
            desc = df_global[column].describe()
            stats = {
                'type': 'numerical',
                'count': int(desc['count']),
                'mean': round(desc['mean'], 2),
                'std': round(desc['std'], 2),
                'min': round(desc['min'], 2),
                '25%': round(desc['25%'], 2),
                '50%': round(desc['50%'], 2),
                '75%': round(desc['75%'], 2),
                'max': round(desc['max'], 2),
                'unique': df_global[column].nunique(),
                'missing': df_global[column].isnull().sum(),
                'skewness': round(df_global[column].skew(), 2),
                'kurtosis': round(df_global[column].kurtosis(), 2)
            }
        else:
            # Categorical column statistics
            value_counts = df_global[column].value_counts()
            stats = {
                'type': 'categorical',
                'count': len(df_global[column]),
                'unique': df_global[column].nunique(),
                'missing': df_global[column].isnull().sum(),
                'top_categories': value_counts.head().to_dict()
            }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_csv')
def download_csv():
    global df_global
    if df_global is not None:
        csv_buffer = io.StringIO()
        df_global.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='analyzed_data.csv'
        )
    return "No file to download!"

@app.route('/download_plot/<plot_type>')
def download_plot(plot_type):
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{plot_type}.png')
    if os.path.exists(plot_path):
        return send_file(plot_path, as_attachment=True)
    return "Plot not found!"

def generate_visualizations(df, numeric_cols, categorical_cols):
    plot_files = {}
    
    # Histograms for numeric columns
    if numeric_cols:
        plt.figure(figsize=(12, 8))
        df[numeric_cols].hist(bins=20, grid=False)
        plt.tight_layout()
        hist_path = os.path.join(app.config['UPLOAD_FOLDER'], 'histograms.png')
        plt.savefig(hist_path)
        plt.close()
        plot_files['histograms'] = hist_path
    
    # Box plots for numeric columns
    if numeric_cols:
        plt.figure(figsize=(12, 8))
        df[numeric_cols].plot(kind='box', subplots=True, layout=(1, len(numeric_cols)), 
                             sharex=False, sharey=False)
        plt.tight_layout()
        box_path = os.path.join(app.config['UPLOAD_FOLDER'], 'boxplots.png')
        plt.savefig(box_path)
        plt.close()
        plot_files['boxplots'] = box_path
    
    # Count plots for categorical columns (first 5)
    if categorical_cols:
        for i, col in enumerate(categorical_cols[:5]):
            plt.figure(figsize=(10, 6))
            value_counts = df[col].value_counts()
            if len(value_counts) > 10:
                # For columns with many categories, show top 10
                value_counts = value_counts.head(10)
            plt.bar(value_counts.index.astype(str), value_counts.values)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            cat_path = os.path.join(app.config['UPLOAD_FOLDER'], f'category_{col}.png')
            plt.savefig(cat_path)
            plt.close()
            plot_files[f'category_{col}'] = cat_path
    
    return plot_files

if __name__ == '__main__':
    app.run(debug=True)