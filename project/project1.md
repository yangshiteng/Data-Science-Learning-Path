            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize
            from sklearn.decomposition import TruncatedSVD
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            import string
            
            # Assuming texts is your list of metes and bounds legal descriptions
            texts = ["Text 1", "Text 2", "Text 3", "..."]
            
            # Download necessary NLTK data
            nltk.download(['punkt', 'wordnet', 'stopwords'])
            
            # Initialize a lemmatizer
            lemmatizer = WordNetLemmatizer()
            
            # Define a function to preprocess the texts
            def preprocess(text):
            # Tokenize and remove punctuation
            words = nltk.word_tokenize(text.translate(str.maketrans('', '', string.punctuation)))
            # Remove stopwords and lemmatize
            return [lemmatizer.lemmatize(word.lower()) for word in words if word not in stopwords.words('english')]
            
            # Preprocess the texts
            processed_texts = [' '.join(preprocess(text)) for text in texts]
            
            # Use TF-IDF to transform the texts into vectors
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(processed_texts)
            
            # Normalize the vectors
            X = normalize(X)
            
            # Use TruncatedSVD to reduce dimensionality
            svd = TruncatedSVD(n_components=50)
            X = svd.fit_transform(X)
            
            # Decide on the number of clusters
            n_clusters = 5
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(X)
            
            # Print the cluster assignments
            print(kmeans.labels_)


BEGINNING AT A TURNING POINT IN THE CENTER OF STATE ROAD #2213 SAID POINT BEING THE SOUTHWESTERN MOST CORNER OF LOT 5 AND THE SOUTHEASTERN MOST CORNER OF LOT #4 AS SHOWN UPON THAT SURVEY RECORDED IN PLAT BOOK 14 AT PAGE 105, RCR; AND RUNS THENCE FROM SAID BEGINNING POINT WITH THE COMMON LINE OF LOTS 4 AND 5 NORTH 08 DEGREES 35 MINUTES 13 SECONDS EAST 145.00 FEET TO AN EXISTING IRON PIN (SAID LINE CROSSING AN EXISTING IRON PIN AT 32.20 FEET); THENCE SOUTH 81 DEGREES 35 MINUTES 13 SECONDS EAST 145 FEET TO AN EXISTING IRON PIN; THENCE SOUTH 08 DEGREES 24 MINUTES 47 SECONDS WEST 229.50 FEET TO A TURNING POINT IN THE CENTER LINE OF MCDADE ROAD (SAID LINE CROSSING AN EXISTING IRON PIN AT 184.80 FEET; THENCE THE FOLLOWING 2 CALLS WITH THE CENTERLINE OF SAID ROAD; NORTH 70 DEGREES 03 MINUTES 55 SECONDS WEST 40.09 FEET; NORTH 71 DEGREES 42 MINUTES 11 SECONDS WEST 68.58 FEET; THENCE A LINE NORTH 17 DEGREES 00 MINUTES 00 SECONDS EAST 9.83 FEET TO A TURNING POINT NEAR THE NORTH EDGE OF MCDADE ROAD; THENCE NORTH 81 DEGREES 35 MINUTES 13 SECONDS WEST 39.62 FEET TO THE POINT AND PLACE OF BEGINNING, SAID LOT CONTAINING 0.71 ACRES MORE OR LESS.
import math

# List of moves
moves = [
    ("N", 8, 35, 13, 145.00),
    ("E", 81, 35, 13, 145),
    ("S", 8, 24, 47, 229.50),
    ("W", 70, 3, 55, 40.09),
    ("W", 71, 42, 11, 68.58),
    ("N", 17, 0, 0, 9.83),
    ("W", 81, 35, 13, 39.62)
]

# Start point
x, y = 0, 0

# List of points
points = [(x, y)]

# Process each move
for move in moves:
    direction = move[0]
    degrees = move[1] + move[2] / 60 + move[3] / 3600
    distance = move[-1]
    
    # Convert degrees to radians
    angle = math.radians(degrees)

    if direction in {"N", "S"}:
        angle = math.pi / 2 - angle
    if direction in {"S", "W"}:
        angle += math.pi
    # Calculate displacement
    dx = distance * math.cos(angle)
    dy = distance * math.sin(angle)
    # Update coordinates
    x += dx
    y += dy
    points.append((x, y))

# Check if the first and last points are the same
is_closed = points[0] == points[-1]

print("The property is", "enclosed" if is_closed else "not enclosed")

