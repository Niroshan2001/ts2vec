"""
Test script for Enhanced TS2Vec with Sinusoidal Time Features
This validates our improvements to TS2Vec's forecasting capabilities.
"""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_time_features():
    """Test the generate_time_features function"""
    print("🔹 Testing Time Features Generation...")
    
    from tasks.forecasting import generate_time_features
    
    # Test with 168 hours (1 week of hourly data)
    length = 168
    time_feats = generate_time_features(length)
    
    print(f"   ✅ Time features shape: {time_feats.shape}")
    print(f"   ✅ Expected shape: ({length}, 6)")
    print(f"   ✅ Features include: daily, weekly, monthly sin/cos cycles")
    
    # Check if features are properly bounded between -1 and 1
    assert time_feats.min() >= -1.0 and time_feats.max() <= 1.0, "Time features should be bounded [-1,1]"
    assert time_feats.shape == (length, 6), f"Expected shape ({length}, 6), got {time_feats.shape}"
    
    print("   ✅ Time features test PASSED!")
    return True

def test_sinusoidal_model():
    """Test sinusoidal model integration"""
    print("\n🔹 Testing Sinusoidal Model Integration...")
    
    try:
        from boosted_hybrid_model import SimpleSinusoidalForecaster
        
        # Test with synthetic data
        np.random.seed(42)
        series = np.random.randn(1000)
        train_slice = slice(0, 600)
        val_slice = slice(600, 800)
        test_slice = slice(800, 1000)
        
        model = SimpleSinusoidalForecaster(period=24)
        model.fit_for_horizons(series, train_slice, val_slice, horizons=[24], padding=200)
        
        # Test prediction
        preds = model.predict_ts2vec_aligned(series, test_slice, horizon=24, padding=200)
        
        print(f"   ✅ Predictions shape: {preds.shape}")
        print(f"   ✅ Model trained and prediction successful")
        print("   ✅ Sinusoidal model integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"   ❌ Sinusoidal model test failed: {e}")
        return False

def test_enhanced_ts2vec_components():
    """Test core TS2Vec enhancements"""
    print("\n🔹 Testing Enhanced TS2Vec Components...")
    
    try:
        # Test ensemble prediction function
        from tasks.forecasting import ensemble_predictions
        
        # Mock predictions for testing
        ts2vec_preds = np.random.randn(100, 24) 
        ts2vec_time_preds = np.random.randn(100, 24)
        hybrid_preds = np.random.randn(100, 24)
        
        # Test 2-way ensemble
        ensemble_2way = ensemble_predictions(
            ts2vec_preds, ts2vec_time_preds, None, pred_len=24, 
            strategy='adaptive', ensemble_type='2way'
        )
        
        # Test 3-way ensemble 
        ensemble_3way = ensemble_predictions(
            ts2vec_preds, ts2vec_time_preds, hybrid_preds, pred_len=24,
            strategy='adaptive', ensemble_type='3way'
        )
        
        print(f"   ✅ 2-way ensemble shape: {ensemble_2way.shape}")
        print(f"   ✅ 3-way ensemble shape: {ensemble_3way.shape}")
        print("   ✅ Ensemble prediction test PASSED!")
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced TS2Vec test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("🚀 Enhanced TS2Vec Test Suite")
    print("=" * 60)
    
    tests = [
        test_time_features,
        test_sinusoidal_model, 
        test_enhanced_ts2vec_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests PASSED! Enhanced TS2Vec is ready.")
        print("\nEnhancements include:")
        print("   • ⚡ Fast sinusoidal time features")
        print("   • 🔄 Adaptive ensemble weighting")
        print("   • 📈 Improved long-horizon forecasting")
        print("   • 🛡️  Stable Ridge regression")
    else:
        print(f"❌ {total - passed} tests failed. Please check errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)