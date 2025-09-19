from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class FinanceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    month = db.Column(db.String(20), nullable=False)
    revenue = db.Column(db.Float, nullable=False)
    expenses = db.Column(db.Float, nullable=False)
    profit = db.Column(db.Float, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "month": self.month,
            "revenue": self.revenue,
            "expenses": self.expenses,
            "profit": self.profit
        }
