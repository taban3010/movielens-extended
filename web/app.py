"""
Flask Web Application for Movie Recommendation System
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import sys
import os
import io
import base64
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the parent directory to the path to access src module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import CollaborativeFilteringRecommender, RecommenderVisualizer, DataGenerator, FileManager

app = Flask(__name__)
app.secret_key = 'movie_recommender_secret_key_2025'

# Global storage for session data (in production, use proper session storage)
session_data = {}


@app.route('/')
def index():
    """Main page of the application."""
    return render_template('index.html')


@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    """Generate a synthetic dataset based on user parameters."""
    try:
        # Get parameters from form
        dataset_type = request.form.get('dataset_type', 'simple')
        n_users = int(request.form.get('n_users', 20))
        n_movies = int(request.form.get('n_movies', 50))
        seed = request.form.get('seed')
        
        # Convert empty seed to None
        if seed:
            seed = int(seed)
        else:
            seed = None
        
        # Type-specific parameters
        params = {
            'dataset_type': dataset_type,
            'n_users': n_users,
            'n_movies': n_movies
        }
        
        if dataset_type == 'realistic':
            params.update({
                'sparsity': float(request.form.get('sparsity', 0.95))
            })
        elif dataset_type == 'clustered':
            params.update({
                'n_clusters': int(request.form.get('n_clusters', 4))
            })
        elif dataset_type == 'simple':
            params.update({
                'min_ratings_per_user': int(request.form.get('min_ratings_per_user', 2)),
                'max_ratings_per_user': int(request.form.get('max_ratings_per_user', 8))
            })
        
        # Generate dataset
        generator = DataGenerator(seed=seed)
        dataset, metadata = generator.generate_dataset_with_parameters(**params)
        
        # Store in session data
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_data[session_id] = {
            'dataset': dataset,
            'metadata': metadata,
            'recommender': None,
            'recommendations': {},
            'visualizations': {}
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'statistics': metadata['statistics'],
            'message': f'Dataset generated successfully with {len(dataset)} users'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Run collaborative filtering analysis on the generated dataset."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        algorithm_type = data.get('algorithm_type', 'original')
        target_user = int(data.get('target_user', 0))
        n_recommendations = int(data.get('n_recommendations', 5))
        
        # Original algorithm parameters
        similarity_method = data.get('similarity_method', 'cosine')
        recommendation_method = data.get('recommendation_method', 'user_based')
        k_neighbors = int(data.get('k_neighbors', 5))
        
        # Extended algorithm parameters
        n_factors = int(data.get('n_factors', 20))
        
        if session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        session = session_data[session_id]
        dataset = session['dataset']
        
        # Initialize recommender if not exists
        if session['recommender'] is None:
            recommender = CollaborativeFilteringRecommender()
            recommender.load_data(dataset)
            session['recommender'] = recommender
        else:
            recommender = session['recommender']
        
        # Generate recommendations based on algorithm type
        if algorithm_type == 'original':
            # Calculate similarities for original algorithm
            recommender.calculate_user_similarity(method=similarity_method)
            recommender.calculate_item_similarity(method=similarity_method)
            
            # Generate recommendations using collaborative filtering
            recommendations = recommender.get_recommendations(
                target_user, 
                method=recommendation_method,
                n_recommendations=n_recommendations,
                k=k_neighbors
            )
            
            method_description = f"{recommendation_method.replace('_', '-').title()} Collaborative Filtering"
            
        else:  # extended algorithm
            # Generate recommendations using matrix factorization
            recommendations = recommender.extended_algorithm(
                target_user,
                n_recommendations=n_recommendations,
                n_factors=n_factors
            )
            
            method_description = f"Matrix Factorization (SVD with {n_factors} factors)"
        
        # Store recommendations
        session['recommendations'][f'{target_user}_{algorithm_type}'] = recommendations
        session['metadata']['algorithm_type'] = algorithm_type
        session['metadata']['last_analysis'] = {
            'target_user': target_user,
            'algorithm_type': algorithm_type,
            'method_description': method_description,
            'n_recommendations': n_recommendations
        }
        
        # Format recommendations for response
        rec_list = [
            {'movie_id': movie_id, 'predicted_rating': round(rating, 2)}
            for movie_id, rating in recommendations
        ]
        
        return jsonify({
            'success': True,
            'recommendations': rec_list,
            'user_id': target_user,
            'algorithm_type': algorithm_type,
            'method_description': method_description,
            'message': f'Generated {len(recommendations)} recommendations using {method_description}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/generate_visualization', methods=['POST'])
def generate_visualization():
    """Generate visualization plots for the analysis."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        plot_type = data.get('plot_type', 'analysis')
        
        if session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        session = session_data[session_id]
        recommender = session['recommender']
        
        if recommender is None:
            return jsonify({'success': False, 'error': 'Analysis not run yet'}), 400
        
        visualizer = RecommenderVisualizer(recommender)
        
        if plot_type == 'analysis':
            # Generate data analysis plot
            fig = visualizer.plot_data_analysis()
            
        elif plot_type == 'recommendations':
            # Generate recommendations plot
            target_user = data.get('target_user', 0)
            method = data.get('method', 'user_based')
            rec_key = f'{target_user}_{method}'
            
            if rec_key not in session['recommendations']:
                return jsonify({'success': False, 'error': 'Recommendations not found'}), 404
            
            recommendations = session['recommendations'][rec_key]
            fig = visualizer.plot_recommendations(target_user, recommendations)
        
        else:
            return jsonify({'success': False, 'error': 'Unknown plot type'}), 400
        
        # Convert plot to base64
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        plt.close(fig)  # Clean up
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'plot_type': plot_type
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/download_results', methods=['POST'])
def download_results():
    """Create and provide a downloadable package of all results."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        session = session_data[session_id]
        
        # Create file manager
        file_manager = FileManager(base_path=f"outputs/{session_id}")
        
        # Create download package
        package_path = file_manager.create_download_package(
            dataset=session['dataset'],
            recommendations=session['recommendations'],
            metadata=session['metadata'],
            package_name=f"movie_recommendations_{session_id}"
        )
        
        # Convert to absolute path for Flask's send_file
        absolute_path = os.path.abspath(package_path)
        
        return send_file(
            absolute_path,
            as_attachment=True,
            download_name=f"movie_recommendations_{session_id}.zip",
            mimetype='application/zip'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/get_user_info/<session_id>/<int:user_id>')
def get_user_info(session_id, user_id):
    """Get information about a specific user's ratings."""
    try:
        if session_id not in session_data:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        session = session_data[session_id]
        dataset = session['dataset']
        
        # Find user in dataset
        user_data = None
        for user in dataset:
            if int(user['user_id']) == user_id:
                user_data = user
                break
        
        if user_data is None:
            return jsonify({'success': False, 'error': f'User {user_id} not found'}), 404
        
        ratings = user_data['ratings']
        avg_rating = sum(r['rating'] for r in ratings) / len(ratings) if ratings else 0
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'total_ratings': len(ratings),
            'average_rating': round(avg_rating, 2),
            'ratings': ratings
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/sessions')
def list_sessions():
    """List all active sessions."""
    sessions = []
    for session_id, data in session_data.items():
        sessions.append({
            'session_id': session_id,
            'statistics': data['metadata']['statistics'],
            'has_analysis': data['recommender'] is not None,
            'num_recommendations': len(data['recommendations'])
        })
    
    return jsonify({
        'success': True,
        'sessions': sessions
    })


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
