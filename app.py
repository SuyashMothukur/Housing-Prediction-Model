from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load model at startup
print("Loading model...")
with open('xgboost_housing_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Load training data for target encoding lookups
training_data = pd.read_csv('housing_data_cleaned.csv')

# Pre-calculate target encoding mappings
city_encoding = training_data.groupby('city')['price'].mean().to_dict()
state_encoding = training_data.groupby('state')['price'].mean().to_dict()
zip_encoding = training_data.groupby('zip_code')['price'].mean().to_dict()

# Region mapping
region_mapping = {
    'Connecticut': 'Northeast', 'Maine': 'Northeast', 'Massachusetts': 'Northeast',
    'New Hampshire': 'Northeast', 'Rhode Island': 'Northeast', 'Vermont': 'Northeast',
    'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast',
    'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Michigan': 'Midwest',
    'Ohio': 'Midwest', 'Wisconsin': 'Midwest', 'Iowa': 'Midwest',
    'Kansas': 'Midwest', 'Minnesota': 'Midwest', 'Missouri': 'Midwest',
    'Nebraska': 'Midwest', 'North Dakota': 'Midwest', 'South Dakota': 'Midwest',
    'Delaware': 'South', 'Florida': 'South', 'Georgia': 'South',
    'Maryland': 'South', 'North Carolina': 'South', 'South Carolina': 'South',
    'Virginia': 'South', 'District of Columbia': 'South', 'West Virginia': 'South',
    'Alabama': 'South', 'Kentucky': 'South', 'Mississippi': 'South',
    'Tennessee': 'South', 'Arkansas': 'South', 'Louisiana': 'South',
    'Oklahoma': 'South', 'Texas': 'South',
    'Arizona': 'West', 'Colorado': 'West', 'Idaho': 'West',
    'Montana': 'West', 'Nevada': 'West', 'New Mexico': 'West',
    'Utah': 'West', 'Wyoming': 'West', 'Alaska': 'West',
    'California': 'West', 'Hawaii': 'West', 'Oregon': 'West',
    'Washington': 'West',
    'Puerto Rico': 'South', 'Virgin Islands': 'South'
}

region_encoding = training_data.groupby('region')['price'].mean().to_dict()

print("✓ Model loaded successfully!")
print(f"✓ Loaded {len(city_encoding)} cities")
print(f"✓ Loaded {len(state_encoding)} states")
print(f"✓ Loaded {len(zip_encoding)} zip codes")


def prepare_features(property_data):
    """
    Prepare property data for prediction with TARGET ENCODING
    """
    df = pd.DataFrame([property_data])
    
    city = property_data['city']
    state = property_data['state']
    zip_code = property_data['zip_code']
    
    # ========================================
    # BASIC FEATURES
    # ========================================
    df['price_per_sqft'] = 0  # Placeholder
    df['total_rooms'] = df['bed'] + df['bath']
    df['bath_bed_ratio'] = df['bath'] / df['bed']
    df['size_per_bedroom'] = df['house_size'] / df['bed']
    
    # ========================================
    # TARGET ENCODING (THE FIX!)
    # ========================================
    
    # City target encoding
    df['city_target_encoded'] = city_encoding.get(city, None)
    if df['city_target_encoded'].iloc[0] is None:
        # City not found, use state average
        df['city_target_encoded'] = state_encoding.get(state, training_data['price'].mean())
    
    # State target encoding
    df['state_target_encoded'] = state_encoding.get(state, training_data['price'].mean())
    
    # Zip target encoding
    df['zip_target_encoded'] = zip_encoding.get(zip_code, None)
    if df['zip_target_encoded'].iloc[0] is None:
        # Zip not found, use city encoding
        df['zip_target_encoded'] = df['city_target_encoded']
    
    # Region encoding
    region = region_mapping.get(state, 'Other')
    df['region_target_encoded'] = region_encoding.get(region, training_data['price'].mean())
    
    # State median price
    state_data = training_data[training_data['state'] == state]
    if len(state_data) > 0:
        df['state_median_price'] = state_data['price'].median()
    else:
        df['state_median_price'] = training_data['price'].median()
    
    # ========================================
    # LEGACY FEATURES (for compatibility)
    # ========================================
    city_data = training_data[training_data['city'] == city]
    
    if len(city_data) > 0:
        df['city_median_price'] = city_data['price'].median()
        df['zip_median_ppsf'] = city_data['price_per_sqft'].median()
    else:
        df['city_median_price'] = df['state_median_price']
        df['zip_median_ppsf'] = training_data['price_per_sqft'].median()
    
    df['price_per_sqft'] = df['zip_median_ppsf']
    df['price_vs_city'] = 1.0
    df['price_vs_state'] = 1.0
    
    # ========================================
    # ENSURE ALL FEATURES EXIST
    # ========================================
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select features in correct order
    X = df[feature_names]
    
    return X


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for price predictions
    
    Expected JSON:
    {
        "bed": 3,
        "bath": 2.0,
        "house_size": 2000,
        "city": "Seattle",
        "state": "Washington",
        "zip_code": 98101,
        "acre_lot": 0.15
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['bed', 'bath', 'house_size', 'city', 'state', 'zip_code']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare property data
        property_data = {
            'bed': float(data['bed']),
            'bath': float(data['bath']),
            'house_size': float(data['house_size']),
            'city': str(data['city']),
            'state': str(data['state']),
            'zip_code': int(data['zip_code']),
            'acre_lot': float(data.get('acre_lot', 0))
        }
        
        # Prepare features with TARGET ENCODING
        X = prepare_features(property_data)
        
        # Make prediction
        predicted_price = float(model.predict(X)[0])
        
        # Calculate confidence interval
        mae = metadata['test_mae']
        
        # Get market context for response
        city_avg = city_encoding.get(property_data['city'], state_encoding.get(property_data['state'], 0))
        state_avg = state_encoding.get(property_data['state'], 0)
        
        # Prepare response
        response = {
            'predicted_price': predicted_price,
            'confidence_interval': {
                'lower': predicted_price - mae,
                'upper': predicted_price + mae
            },
            'property_details': property_data,
            'market_context': {
                'city_average': city_avg,
                'state_average': state_avg
            },
            'model_stats': {
                'r2_score': metadata['test_r2'],
                'mae': mae,
                'confidence': metadata['test_r2'] * 100
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_r2': metadata['test_r2'],
        'training_samples': metadata['train_samples']
    }), 200


@app.route('/api/states', methods=['GET'])
def get_states():
    """Get list of available states - includes all 50 US states"""
    # All 50 US states
    all_us_states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
        'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
        'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
        'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
        'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
        'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
        'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
        'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia'
    ]
    
    # Get states from training data (if available)
    training_states = set(training_data['state'].unique().tolist())
    
    # Combine and sort
    all_states = sorted(set(all_us_states) | training_states)
    
    return jsonify({'states': all_states}), 200


@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get cities for a specific state - includes major cities as fallback"""
    state = request.args.get('state')
    if not state:
        return jsonify({'error': 'State parameter required'}), 400
    
    # Major cities by state (comprehensive list)
    major_cities_by_state = {
        'Alabama': ['Birmingham', 'Montgomery', 'Mobile', 'Huntsville', 'Tuscaloosa', 'Hoover', 'Dothan', 'Auburn', 'Decatur', 'Madison'],
        'Alaska': ['Anchorage', 'Fairbanks', 'Juneau', 'Sitka', 'Ketchikan', 'Wasilla', 'Kenai', 'Kodiak'],
        'Arizona': ['Phoenix', 'Tucson', 'Mesa', 'Chandler', 'Scottsdale', 'Glendale', 'Gilbert', 'Tempe', 'Peoria', 'Surprise'],
        'Arkansas': ['Little Rock', 'Fort Smith', 'Fayetteville', 'Springdale', 'Jonesboro', 'North Little Rock', 'Conway', 'Rogers'],
        'California': ['Los Angeles', 'San Diego', 'San Jose', 'San Francisco', 'Fresno', 'Sacramento', 'Long Beach', 'Oakland', 'Bakersfield', 'Anaheim', 'Santa Ana', 'Riverside', 'Stockton', 'Irvine', 'Chula Vista', 'Fremont', 'San Bernardino', 'Modesto', 'Fontana', 'Oxnard', 'Moreno Valley', 'Huntington Beach', 'Glendale', 'Santa Clarita', 'Garden Grove', 'Oceanside', 'Rancho Cucamonga', 'Santa Rosa', 'Ontario', 'Lancaster', 'Elk Grove', 'Corona', 'Palmdale', 'Salinas', 'Pomona', 'Hayward', 'Escondido', 'Torrance', 'Sunnyvale', 'Orange', 'Fullerton', 'Pasadena', 'Thousand Oaks', 'Visalia', 'Simi Valley', 'Concord', 'Roseville', 'Vallejo', 'Victorville', 'Santa Clara', 'Fairfield', 'Inglewood', 'El Monte', 'Richmond', 'Berkeley', 'Downey', 'Antioch', 'Carlsbad', 'Temecula', 'Costa Mesa', 'Murrieta', 'Ventura', 'West Covina', 'El Cajon', 'Daly City', 'Burbank', 'Santa Maria', 'El Centro', 'San Mateo', 'Rialto', 'Mission Viejo', 'Compton', 'South Gate', 'Vista', 'Vacaville', 'Santa Monica', 'Carson', 'Hesperia', 'Santa Barbara', 'Redding', 'Chico', 'San Leandro', 'Livermore', 'Buena Park', 'Hemet', 'Lakewood', 'Merced', 'Napa', 'Redwood City', 'Whittier', 'Hawthorne', 'Citrus Heights', 'Tracy', 'Alhambra', 'Indio', 'Menifee', 'Chino', 'Chino Hills', 'Redondo Beach', 'Newport Beach', 'San Marcos', 'Lake Forest', 'Mountain View', 'Alameda', 'Bellflower', 'Upland', 'Tulare', 'Palo Alto', 'San Rafael', 'Yuba City', 'Folsom', 'Union City', 'Perris', 'Lynwood', 'Apple Valley', 'Redlands', 'Turlock', 'Milpitas', 'Manteca', 'Hanford', 'Pacifica', 'Huntington Park', 'Lodi', 'Madera', 'San Bruno', 'La Habra', 'Watsonville', 'Petaluma', 'San Luis Obispo', 'Davis', 'Camarillo', 'Pittsburg', 'South San Francisco', 'Yucaipa', 'Montebello', 'San Gabriel', 'Highland', 'Hollister', 'Buellton', 'Morgan Hill', 'Seaside', 'Laguna Niguel', 'Lompoc', 'Beaumont', 'Wildomar', 'Los Banos', 'Belmont', 'San Dimas', 'Santee', 'Los Gatos', 'Patterson', 'San Jacinto', 'Brea', 'La Mesa', 'Arcadia', 'Temple City', 'Culver City', 'La Mirada', 'Ceres', 'Calexico', 'Baldwin Park', 'Rancho Palos Verdes', 'Poway', 'La Puente', 'Castro Valley', 'Encinitas', 'National City', 'La Quinta', 'Monterey Park', 'Paramount', 'Cupertino', 'San Ramon', 'La Verne', 'Hemet', 'Foster City', 'Dublin', 'Cypress', 'Monrovia', 'Covina', 'Pico Rivera', 'Rocklin', 'Novato', 'San Pablo', 'Pleasanton', 'Los Altos', 'Fremont', 'San Mateo', 'Folsom', 'Pleasant Hill', 'Walnut Creek', 'Dublin', 'Livermore', 'Union City', 'Hayward', 'Fremont', 'Newark', 'Milpitas', 'Santa Clara', 'Sunnyvale', 'Mountain View', 'Palo Alto', 'Redwood City', 'San Carlos', 'Belmont', 'Foster City', 'Burlingame', 'Millbrae', 'San Bruno', 'South San Francisco', 'Daly City', 'Pacifica', 'Half Moon Bay', 'San Mateo', 'Foster City', 'Belmont', 'San Carlos', 'Redwood City', 'Menlo Park', 'Atherton', 'Portola Valley', 'Woodside', 'Los Altos', 'Los Altos Hills', 'Mountain View', 'Sunnyvale', 'Cupertino', 'Saratoga', 'Los Gatos', 'Monte Sereno', 'Campbell', 'San Jose', 'Milpitas', 'Fremont', 'Newark', 'Union City', 'Hayward', 'San Leandro', 'Alameda', 'Oakland', 'Piedmont', 'Berkeley', 'Albany', 'El Cerrito', 'Richmond', 'San Pablo', 'Hercules', 'Pinole', 'Rodeo', 'Crockett', 'Vallejo', 'Benicia', 'Martinez', 'Pleasant Hill', 'Concord', 'Clayton', 'Pittsburg', 'Antioch', 'Oakley', 'Brentwood', 'Discovery Bay', 'Byron', 'Knightsen', 'Bethel Island', 'Rio Vista', 'Isleton', 'Walnut Grove', 'Courtland', 'Hood', 'Freeport', 'Elk Grove', 'Galt', 'Locke', 'Walnut Grove', 'Courtland', 'Hood', 'Freeport', 'Elk Grove', 'Galt', 'Locke', 'Walnut Grove', 'Courtland', 'Hood', 'Freeport', 'Elk Grove', 'Galt', 'Locke'],
        'Colorado': ['Denver', 'Colorado Springs', 'Aurora', 'Fort Collins', 'Lakewood', 'Thornton', 'Arvada', 'Westminster', 'Pueblo', 'Centennial', 'Boulder', 'Greeley', 'Longmont', 'Loveland', 'Grand Junction', 'Broomfield', 'Commerce City', 'Parker', 'Littleton', 'Northglenn'],
        'Connecticut': ['Bridgeport', 'New Haven', 'Hartford', 'Stamford', 'Waterbury', 'Norwalk', 'Danbury', 'New Britain', 'West Hartford', 'Greenwich'],
        'Delaware': ['Wilmington', 'Dover', 'Newark', 'Middletown', 'Smyrna', 'Milford', 'Seaford', 'Georgetown', 'Elsmere', 'New Castle'],
        'District of Columbia': ['Washington'],
        'Florida': ['Jacksonville', 'Miami', 'Tampa', 'Orlando', 'St. Petersburg', 'Hialeah', 'Tallahassee', 'Fort Lauderdale', 'Port St. Lucie', 'Cape Coral', 'Pembroke Pines', 'Hollywood', 'Miramar', 'Gainesville', 'Coral Springs', 'Miami Gardens', 'Clearwater', 'Palm Bay', 'West Palm Beach', 'Pompano Beach', 'Lakeland', 'Davie', 'Miami Beach', 'Sunrise', 'Plantation', 'Boca Raton', 'Deltona', 'Largo', 'Deerfield Beach', 'Boynton Beach', 'Fort Myers', 'Kissimmee', 'Homestead', 'Tamarac', 'Delray Beach', 'Daytona Beach', 'North Miami', 'Wellington', 'Jupiter', 'Ocala', 'Port Orange', 'Margate', 'Coconut Creek', 'Sanford', 'Sarasota', 'Pensacola', 'Bradenton', 'Palm Coast', 'Fort Pierce', 'Melbourne', 'Coral Gables', 'Key West', 'Naples', 'St. Augustine', 'Gainesville', 'Tallahassee', 'Orlando', 'Tampa', 'Miami', 'Jacksonville', 'Fort Lauderdale', 'West Palm Beach', 'St. Petersburg', 'Clearwater', 'Lakeland', 'Pompano Beach', 'Hollywood', 'Miramar', 'Coral Springs', 'Pembroke Pines', 'Hialeah', 'Miami Gardens', 'Sunrise', 'Plantation', 'Davie', 'Boca Raton', 'Deltona', 'Largo', 'Deerfield Beach', 'Boynton Beach', 'Fort Myers', 'Kissimmee', 'Homestead', 'Tamarac', 'Delray Beach', 'Daytona Beach', 'North Miami', 'Wellington', 'Jupiter', 'Ocala', 'Port Orange', 'Margate', 'Coconut Creek', 'Sanford', 'Sarasota', 'Pensacola', 'Bradenton', 'Palm Coast', 'Fort Pierce', 'Melbourne', 'Coral Gables', 'Key West', 'Naples', 'St. Augustine'],
        'Georgia': ['Atlanta', 'Augusta', 'Columbus', 'Savannah', 'Athens', 'Sandy Springs', 'Roswell', 'Macon', 'Johns Creek', 'Albany', 'Warner Robins', 'Alpharetta', 'Marietta', 'Valdosta', 'Smyrna', 'Dunwoody', 'Rome', 'East Point', 'Peachtree Corners', 'Gainesville'],
        'Hawaii': ['Honolulu', 'Hilo', 'Kailua', 'Kaneohe', 'Kahului', 'Pearl City', 'Waipahu', 'Ewa Beach', 'Mililani', 'Kihei'],
        'Idaho': ['Boise', 'Nampa', 'Meridian', 'Idaho Falls', 'Pocatello', 'Caldwell', 'Coeur d\'Alene', 'Twin Falls', 'Lewiston', 'Post Falls'],
        'Illinois': ['Chicago', 'Aurora', 'Rockford', 'Joliet', 'Naperville', 'Springfield', 'Peoria', 'Elgin', 'Waukegan', 'Cicero', 'Champaign', 'Bloomington', 'Arlington Heights', 'Evanston', 'Schaumburg', 'Bolingbrook', 'Palatine', 'Skokie', 'Des Plaines', 'Orland Park'],
        'Indiana': ['Indianapolis', 'Fort Wayne', 'Evansville', 'South Bend', 'Carmel', 'Fishers', 'Bloomington', 'Hammond', 'Gary', 'Muncie', 'Terre Haute', 'Lafayette', 'Kokomo', 'Anderson', 'Noblesville', 'Greenwood', 'Elkhart', 'Mishawaka', 'Lawrence', 'Jeffersonville'],
        'Iowa': ['Des Moines', 'Cedar Rapids', 'Davenport', 'Sioux City', 'Iowa City', 'Waterloo', 'Council Bluffs', 'Ames', 'West Des Moines', 'Dubuque'],
        'Kansas': ['Wichita', 'Overland Park', 'Kansas City', 'Olathe', 'Topeka', 'Lawrence', 'Shawnee', 'Manhattan', 'Lenexa', 'Salina'],
        'Kentucky': ['Louisville', 'Lexington', 'Bowling Green', 'Owensboro', 'Covington', 'Hopkinsville', 'Richmond', 'Florence', 'Georgetown', 'Henderson'],
        'Louisiana': ['New Orleans', 'Baton Rouge', 'Shreveport', 'Lafayette', 'Lake Charles', 'Kenner', 'Bossier City', 'Monroe', 'Alexandria', 'Houma'],
        'Maine': ['Portland', 'Lewiston', 'Bangor', 'South Portland', 'Auburn', 'Biddeford', 'Sanford', 'Saco', 'Augusta', 'Westbrook'],
        'Maryland': ['Baltimore', 'Frederick', 'Rockville', 'Gaithersburg', 'Bowie', 'Annapolis', 'College Park', 'Salisbury', 'Laurel', 'Greenbelt'],
        'Massachusetts': ['Boston', 'Worcester', 'Springfield', 'Lowell', 'Cambridge', 'New Bedford', 'Brockton', 'Quincy', 'Lynn', 'Fall River'],
        'Michigan': ['Detroit', 'Grand Rapids', 'Warren', 'Sterling Heights', 'Lansing', 'Ann Arbor', 'Flint', 'Dearborn', 'Livonia', 'Troy'],
        'Minnesota': ['Minneapolis', 'St. Paul', 'Rochester', 'Duluth', 'Bloomington', 'Brooklyn Park', 'Plymouth', 'St. Cloud', 'Eagan', 'Woodbury'],
        'Mississippi': ['Jackson', 'Gulfport', 'Southaven', 'Hattiesburg', 'Biloxi', 'Meridian', 'Tupelo', 'Greenville', 'Olive Branch', 'Horn Lake'],
        'Missouri': ['Kansas City', 'St. Louis', 'Springfield', 'Columbia', 'Independence', 'Lee\'s Summit', 'O\'Fallon', 'St. Joseph', 'St. Charles', 'St. Peters'],
        'Montana': ['Billings', 'Missoula', 'Great Falls', 'Bozeman', 'Butte', 'Helena', 'Kalispell', 'Havre', 'Anaconda', 'Miles City'],
        'Nebraska': ['Omaha', 'Lincoln', 'Bellevue', 'Grand Island', 'Kearney', 'Fremont', 'Hastings', 'North Platte', 'Norfolk', 'Columbus'],
        'Nevada': ['Las Vegas', 'Henderson', 'Reno', 'North Las Vegas', 'Sparks', 'Carson City', 'Fernley', 'Elko', 'Mesquite', 'Boulder City'],
        'New Hampshire': ['Manchester', 'Nashua', 'Concord', 'Derry', 'Rochester', 'Dover', 'Salem', 'Merrimack', 'Londonderry', 'Hudson'],
        'New Jersey': ['Newark', 'Jersey City', 'Paterson', 'Elizabeth', 'Edison', 'Woodbridge', 'Lakewood', 'Toms River', 'Hamilton', 'Trenton'],
        'New Mexico': ['Albuquerque', 'Las Cruces', 'Rio Rancho', 'Santa Fe', 'Roswell', 'Farmington', 'Clovis', 'Hobbs', 'Alamogordo', 'Carlsbad'],
        'New York': ['New York', 'Buffalo', 'Rochester', 'Yonkers', 'Syracuse', 'Albany', 'New Rochelle', 'Mount Vernon', 'Schenectady', 'Utica', 'White Plains', 'Hempstead', 'Troy', 'Niagara Falls', 'Binghamton', 'Freeport', 'Valley Stream', 'Long Beach', 'Rome', 'Ithaca'],
        'North Carolina': ['Charlotte', 'Raleigh', 'Greensboro', 'Durham', 'Winston-Salem', 'Fayetteville', 'Cary', 'Wilmington', 'High Point', 'Concord'],
        'North Dakota': ['Fargo', 'Bismarck', 'Grand Forks', 'Minot', 'West Fargo', 'Williston', 'Dickinson', 'Mandan', 'Jamestown', 'Wahpeton'],
        'Ohio': ['Columbus', 'Cleveland', 'Cincinnati', 'Toledo', 'Akron', 'Dayton', 'Parma', 'Canton', 'Youngstown', 'Lorain'],
        'Oklahoma': ['Oklahoma City', 'Tulsa', 'Norman', 'Broken Arrow', 'Lawton', 'Edmond', 'Moore', 'Midwest City', 'Enid', 'Stillwater'],
        'Oregon': ['Portland', 'Eugene', 'Salem', 'Gresham', 'Hillsboro', 'Bend', 'Beaverton', 'Medford', 'Springfield', 'Corvallis'],
        'Pennsylvania': ['Philadelphia', 'Pittsburgh', 'Allentown', 'Erie', 'Reading', 'Scranton', 'Bethlehem', 'Lancaster', 'Harrisburg', 'Altoona'],
        'Rhode Island': ['Providence', 'Warwick', 'Cranston', 'Pawtucket', 'East Providence', 'Woonsocket', 'Newport', 'Central Falls', 'Westerly', 'Cumberland'],
        'South Carolina': ['Charleston', 'Columbia', 'North Charleston', 'Mount Pleasant', 'Rock Hill', 'Greenville', 'Summerville', 'Sumter', 'Hilton Head Island', 'Florence'],
        'South Dakota': ['Sioux Falls', 'Rapid City', 'Aberdeen', 'Brookings', 'Watertown', 'Mitchell', 'Yankton', 'Pierre', 'Huron', 'Vermillion'],
        'Tennessee': ['Nashville', 'Memphis', 'Knoxville', 'Chattanooga', 'Clarksville', 'Murfreesboro', 'Franklin', 'Jackson', 'Johnson City', 'Bartlett'],
        'Texas': ['Houston', 'San Antonio', 'Dallas', 'Austin', 'Fort Worth', 'El Paso', 'Arlington', 'Corpus Christi', 'Plano', 'Laredo', 'Lubbock', 'Garland', 'Irving', 'Amarillo', 'Grand Prairie', 'Brownsville', 'McKinney', 'Frisco', 'Pasadena', 'Killeen', 'Mesquite', 'McAllen', 'Carrollton', 'Midland', 'Denton', 'Abilene', 'Beaumont', 'Round Rock', 'Odessa', 'Waco', 'Richardson', 'Lewisville', 'Tyler', 'College Station', 'Pearland', 'Wichita Falls', 'San Angelo', 'League City', 'Longview', 'Sugar Land', 'Edinburg', 'Bryan', 'Baytown', 'Pharr', 'Missouri City', 'Temple', 'Flower Mound', 'Harlingen', 'North Richland Hills', 'Victoria', 'Conroe', 'New Braunfels', 'Mansfield', 'Cedar Park', 'Rowlett', 'Port Arthur', 'Euless', 'Georgetown', 'Pflugerville', 'DeSoto', 'San Marcos', 'Grapevine', 'Bedford', 'Galveston', 'Cedar Hill', 'Texas City', 'Wylie', 'Haltom City', 'Keller', 'Sherman', 'Rockwall', 'Friendswood', 'Mission', 'Bryan', 'Baytown', 'Pharr', 'Missouri City', 'Temple', 'Flower Mound', 'Harlingen', 'North Richland Hills', 'Victoria', 'Conroe', 'New Braunfels', 'Mansfield', 'Cedar Park', 'Rowlett', 'Port Arthur', 'Euless', 'Georgetown', 'Pflugerville', 'DeSoto', 'San Marcos', 'Grapevine', 'Bedford', 'Galveston', 'Cedar Hill', 'Texas City', 'Wylie', 'Haltom City', 'Keller', 'Sherman', 'Rockwall', 'Friendswood', 'Mission'],
        'Utah': ['Salt Lake City', 'West Valley City', 'Provo', 'West Jordan', 'Orem', 'Sandy', 'Ogden', 'St. George', 'Layton', 'Taylorsville'],
        'Vermont': ['Burlington', 'Essex', 'South Burlington', 'Colchester', 'Rutland', 'Montpelier', 'Barre', 'St. Albans', 'Brattleboro', 'Milton'],
        'Virginia': ['Virginia Beach', 'Norfolk', 'Chesapeake', 'Richmond', 'Newport News', 'Alexandria', 'Hampton', 'Portsmouth', 'Suffolk', 'Roanoke'],
        'Washington': ['Seattle', 'Spokane', 'Tacoma', 'Vancouver', 'Bellevue', 'Kent', 'Everett', 'Renton', 'Yakima', 'Federal Way', 'Spokane Valley', 'Bellingham', 'Kennewick', 'Auburn', 'Pasco', 'Marysville', 'Lakewood', 'Redmond', 'Shoreline', 'Richland', 'Kirkland', 'Burien', 'Olympia', 'Lacey', 'Edmonds', 'Bremerton', 'Puyallup', 'Sammamish', 'Bothell', 'Longview', 'Pullman', 'Wenatchee', 'Centralia', 'Mount Vernon', 'Oak Harbor', 'Mukilteo', 'Tumwater', 'Walla Walla', 'University Place', 'Issaquah', 'Washougal', 'Des Moines', 'Lake Stevens', 'Mercer Island', 'Port Angeles', 'Anacortes', 'Snoqualmie', 'Sequim', 'Gig Harbor', 'Port Townsend', 'Enumclaw', 'Snohomish', 'Monroe', 'Sultan', 'Gold Bar', 'Index', 'Skykomish', 'Baring', 'Grotto', 'Scenic', 'Alpine', 'Cascade', 'Leavenworth', 'Cashmere', 'Chelan', 'Manson', 'Stehekin', 'Twisp', 'Winthrop', 'Mazama', 'Conconully', 'Okanogan', 'Omak', 'Tonasket', 'Oroville', 'Loomis', 'Nighthawk', 'Malo', 'Curlew', 'Republic', 'Kettle Falls', 'Colville', 'Chewelah', 'Addy', 'Springdale', 'Valley', 'Deer Park', 'Medical Lake', 'Cheney', 'Airway Heights', 'Spokane', 'Spokane Valley', 'Liberty Lake', 'Post Falls', 'Coeur d\'Alene', 'Hayden', 'Rathdrum', 'Spirit Lake', 'Athol', 'Bayview', 'Careywood', 'Clark Fork', 'Cocolalla', 'Coolin', 'Hope', 'Laclede', 'Nordman', 'Ponderay', 'Priest River', 'Sandpoint', 'Sagle', 'Trestle Creek', 'Troy', 'Bonners Ferry', 'Moyie Springs', 'Naples', 'Eastport'],
        'West Virginia': ['Charleston', 'Huntington', 'Parkersburg', 'Morgantown', 'Wheeling', 'Martinsburg', 'Fairmont', 'Beckley', 'Clarksburg', 'South Charleston'],
        'Wisconsin': ['Milwaukee', 'Madison', 'Green Bay', 'Kenosha', 'Racine', 'Appleton', 'Waukesha', 'Oshkosh', 'Eau Claire', 'Janesville'],
        'Wyoming': ['Cheyenne', 'Casper', 'Laramie', 'Gillette', 'Rock Springs', 'Sheridan', 'Green River', 'Evanston', 'Riverton', 'Jackson']
    }
    
    # Get cities from training data
    training_cities = set()
    if state in training_data['state'].values:
        training_cities = set(training_data[training_data['state'] == state]['city'].unique().tolist())
    
    # Get major cities for the state (fallback)
    major_cities = set(major_cities_by_state.get(state, []))
    
    # Combine both lists (training data cities + major cities)
    all_cities = sorted(list(training_cities | major_cities))
    
    return jsonify({'cities': all_cities}), 200


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 HOUSING PRICE PREDICTOR API")
    print("=" * 70)
    print(f"Model R² Score: {metadata['test_r2']:.4f}")
    print(f"Average Error: ${metadata['test_mae']:,.0f}")
    print(f"Training Samples: {metadata['train_samples']:,}")
    print(f"\n🗺️  Geographic Coverage:")
    print(f"   States: {len(state_encoding)}")
    print(f"   Cities: {len(city_encoding)}")
    print(f"   Zip Codes: {len(zip_encoding)}")
    print("\nAPI Endpoints:")
    print("  POST /api/predict - Get price prediction")
    print("  GET  /api/health  - Check API status")
    print("  GET  /api/states  - Get available states")
    print("  GET  /api/cities?state=X - Get cities in state")
    print("\nStarting server on http://localhost:5001")
    print("=" * 70 + "\n")
    
    app.run(debug=True, port=5001)