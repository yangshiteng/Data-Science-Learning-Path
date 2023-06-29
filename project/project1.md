model = Pipeline(steps=[('vect', TfidfVectorizer(dtype = ny.float32, max_df=0.5, max_features =8000, ngram_range=(1,3))),('svd',TruncatedSVD(n_components=400, random_state=42)), ('scl', StandardScaler()),
('clf',LogisticRegression(C=2, max_iter=200, multi_class='ovr',n_jobs=-1, random_state=42))])
