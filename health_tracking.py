from flask import Blueprint, request, jsonify
from auth import token_required
from models import db, User, Diagnosis

health = Blueprint('health', __name__)

@health.route('/api/v1/health/history', methods=['GET'])
@token_required
def get_health_history(current_user):
    diagnoses = Diagnosis.query.filter_by(user_id=current_user).all()
    return jsonify([{
        'id': d.id,
        'symptoms': d.symptoms,
        'disease': d.predicted_disease,
        'date': d.date_created
    } for d in diagnoses])

@health.route('/api/v1/health/track', methods=['POST'])
@token_required
def track_diagnosis(current_user):
    data = request.get_json()
    new_diagnosis = Diagnosis(
        user_id=current_user,
        symptoms=','.join(data['symptoms']),
        predicted_disease=data['disease']
    )
    db.session.add(new_diagnosis)
    db.session.commit()
    return jsonify({'message': 'Diagnosis tracked successfully'}) 