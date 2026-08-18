# tests/unit/test_synthetic.py
from src.data.synthetic_generator import generate_synthetic_transactions

def test_synthetic_generator_shape():
    df = generate_synthetic_transactions(num_records=100)
    assert len(df) == 100
    assert 'class' in df.columns
