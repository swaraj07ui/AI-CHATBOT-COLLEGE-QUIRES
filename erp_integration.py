"""Simple ERPIntegration stub used by `chatbot.py`.

This module provides a minimal in-memory implementation of the methods
the chatbot expects so the app can run end-to-end without a production ERP.
Replace or extend these methods to integrate with a real ERP system.
"""
from typing import Optional, Dict, Any


class ERPIntegration:
    """Stubbed ERP integration returning sample data.

    The methods return realistic sample dictionaries matching what
    `chatbot.py` expects. Return `None` when no data is available for
    the provided `user_id`.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Placeholder: could load config or credentials from file/env
        self.config = config or {}

    def get_attendance(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        return {
            'overall': '85%',
            'subjects': {
                'Mathematics': '90%',
                'Physics': '80%',
                'Computer Science': '88%'
            }
        }

    def get_results(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        return {
            'current_semester': '4',
            'sgpa': '9.00',
            'cgpa': '8.50',
            'grades': {
                'Mathematics': 'A',
                'Physics': 'B+',
                'Computer Science': 'A-'
            }
        }

    def get_fee_details(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        return {
            'total_fee': '100000',
            'paid': '80000',
            'pending': '20000',
            'due_date': '2025-12-31'
        }

    def get_timetable(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        return {
            'monday': [
                {'time': '09:00-10:00', 'subject': 'Mathematics', 'room': '101'},
                {'time': '10:15-11:15', 'subject': 'Physics', 'room': '102'}
            ],
            'tuesday': [
                {'time': '09:00-10:00', 'subject': 'Computer Science', 'room': '201'},
                {'time': '10:15-11:15', 'subject': 'Mathematics', 'room': '101'}
            ],
            'wednesday': [
                {'time': '09:00-10:00', 'subject': 'Physics', 'room': '102'},
                {'time': '10:15-11:15', 'subject': 'Computer Science', 'room': '201'}
            ]
        }
