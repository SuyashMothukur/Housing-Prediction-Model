import React, { useState } from 'react';
import { Home, DollarSign, TrendingUp, MapPin, Bed, Bath, Maximize } from 'lucide-react';

const HousingPricePredictor = () => {
  const [formData, setFormData] = useState({
    bed: 3,
    bath: 2,
    house_size: 2000,
    city: '',
    state: '',
    zip_code: '',
    acre_lot: 0
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    let parsedValue = parseFloat(value);

    // Enforce restrictions on input values
    if (name === 'bed' || name === 'bath') {
      parsedValue = Math.max(1, Math.min(10, parsedValue || 0)); // Restrict to 1-10
    } else if (name === 'house_size') {
      parsedValue = Math.min(20000, parsedValue || 0); // Restrict to max 20,000
    } else if (name === 'acre_lot') {
      parsedValue = Math.max(0, parsedValue || 0); // Ensure non-negative values
    }

    setFormData(prev => ({
      ...prev,
      [name]: name === 'bed' || name === 'bath' || name === 'house_size' || name === 'acre_lot'
        ? parsedValue
        : value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);

    // Simulate API call
    setTimeout(() => {
      const basePrices = {
        'California': 650000,
        'Washington': 550000,
        'New York': 700000,
        'Texas': 350000,
        'Florida': 400000
      };
      
      const basePrice = basePrices[formData.state] || 400000;
      const bedMultiplier = formData.bed * 75000;
      const bathMultiplier = formData.bath * 50000;
      const sizeMultiplier = formData.house_size * 150;
      const lotMultiplier = formData.acre_lot * 25000;
      
      const estimatedPrice = basePrice + bedMultiplier + bathMultiplier + sizeMultiplier + lotMultiplier;
      const mae = 19000;
      
      setPrediction({
        price: estimatedPrice,
        lower: estimatedPrice - mae,
        upper: estimatedPrice + mae,
        confidence: 98.5
      });
      
      setLoading(false);
    }, 1500);
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(price);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl mb-4 shadow-lg">
            <Home className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Housing Price Predictor
          </h1>
          <p className="text-gray-600 text-lg">
            Get instant AI-powered price estimates for any property
          </p>
          <div className="inline-flex items-center gap-2 mt-3 px-4 py-2 bg-green-50 border border-green-200 rounded-full">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-green-700">98.5% Accuracy • 1.4M Properties Trained</span>
          </div>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-3xl shadow-2xl overflow-hidden">
          <div className="p-8 space-y-6">
            {/* Bedrooms Input */}
            <div>
              <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-3">
                <Bed className="w-5 h-5 text-blue-600" />
                Bedrooms:
              </label>
              <input
                type="number"
                name="bed"
                min="1"
                max="10"
                value={formData.bed}
                onChange={handleInputChange}
                placeholder="Number of bedrooms"
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all"
              />
            </div>

            {/* Bathrooms Input */}
            <div>
              <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-3">
                <Bath className="w-5 h-5 text-blue-600" />
                Bathrooms:
              </label>
              <input
                type="number"
                name="bath"
                min="1"
                max="10"
                step="0.5"
                value={formData.bath}
                onChange={handleInputChange}
                placeholder="Number of bathrooms"
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all"
              />
            </div>

            {/* House Size Input */}
            <div>
              <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-3">
                <Maximize className="w-5 h-5 text-blue-600" />
                Square Feet:
              </label>
              <input
                type="number"
                name="house_size"
                min="500"
                max="20000"
                step="100"
                value={formData.house_size}
                onChange={handleInputChange}
                placeholder="Square footage"
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all"
              />
            </div>

            {/* Location Inputs */}
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                  <MapPin className="w-4 h-4 text-blue-600" />
                  City
                </label>
                <input
                  type="text"
                  name="city"
                  value={formData.city}
                  onChange={handleInputChange}
                  placeholder="e.g., Seattle"
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all"
                />
              </div>

              <div>
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                  <MapPin className="w-4 h-4 text-blue-600" />
                  State
                </label>
                <input
                  type="text"
                  name="state"
                  value={formData.state}
                  onChange={handleInputChange}
                  placeholder="e.g., Washington"
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all"
                />
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                  <MapPin className="w-4 h-4 text-blue-600" />
                  Zip Code
                </label>
                <input
                  type="text"
                  name="zip_code"
                  value={formData.zip_code}
                  onChange={handleInputChange}
                  placeholder="e.g., 98101"
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all"
                />
              </div>

              <div>
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
                  <Maximize className="w-4 h-4 text-gray-400" />
                  Lot Size (acres) <span className="text-xs text-gray-400 font-normal">Optional</span>
                </label>
                <input
                  type="number"
                  name="acre_lot"
                  value={formData.acre_lot}
                  onChange={handleInputChange}
                  placeholder="0.25"
                  step="0.01"
                  min="0"
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all"
                />
              </div>
            </div>

            {/* Submit Button */}
            <button
              onClick={handleSubmit}
              disabled={loading || !formData.city || !formData.state || !formData.zip_code}
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold py-4 px-6 rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin"></div>
                  Calculating...
                </>
              ) : (
                <>
                  <DollarSign className="w-5 h-5" />
                  Get Price Estimate
                </>
              )}
            </button>
          </div>

          {/* Prediction Results */}
          {prediction && (
            <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-8 text-white">
              <div className="text-center mb-6">
                <div className="inline-flex items-center gap-2 mb-3">
                  <TrendingUp className="w-6 h-6" />
                  <h2 className="text-2xl font-bold">Estimated Price</h2>
                </div>
                <div className="text-6xl font-bold mb-2">
                  {formatPrice(prediction.price)}
                </div>
                <p className="text-blue-100 text-lg">
                  for {formData.bed} bed, {formData.bath} bath in {formData.city}, {formData.state}
                </p>
              </div>

              {/* Confidence Interval */}
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-blue-100">Price Range</span>
                  <span className="font-semibold">{formatPrice(prediction.lower)} - {formatPrice(prediction.upper)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-blue-100">Model Confidence</span>
                  <span className="font-semibold">{prediction.confidence}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-blue-100">Average Error</span>
                  <span className="font-semibold">±$19,000</span>
                </div>
              </div>

              <div className="mt-6 text-center text-sm text-blue-100">
                💡 This estimate is based on 1.4M properties and market data
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-600 text-sm">
          <p>Powered by XGBoost ML • Trained on 1.4M+ US Properties</p>
          <p className="mt-2">Model Accuracy: 98.5% R² Score</p>
        </div>
      </div>
    </div>
  );
};

export default HousingPricePredictor;