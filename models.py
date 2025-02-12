from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Diagnosis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    symptoms = db.Column(db.String(500))
    predicted_disease = db.Column(db.String(100))
    confidence_level = db.Column(db.String(20))
    description = db.Column(db.Text)
    ml_prediction = db.Column(db.String(100))
    llm_prediction = db.Column(db.String(100))

    def to_dict(self):
        return {
            'id': self.id,
            'date_created': self.date_created.strftime('%Y-%m-%d %H:%M:%S'),
            'symptoms': self.symptoms,
            'predicted_disease': self.predicted_disease,
            'confidence_level': self.confidence_level,
            'description': self.description,
            'ml_prediction': self.ml_prediction,
            'llm_prediction': self.llm_prediction
        } 