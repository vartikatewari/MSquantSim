def run_random_forest(data, sample_size, rdata, replace):
    # Assume 'condition' is the target variable column in your CSV
    X = data.drop('Condition', axis=1)
    y = data['Condition']
    accuracies = []
    errors = []
    min_accuracies = []
    max_accuracies = []
    #     filepath = 'crc/'
    #     rdata = pd.read_csv(filepath+'crc_silvia_data.csv')
    #     condition_mapping = {'Healthy': 0, 'CRC': 1}
    #     # Replace values in the "Condition" column
    #     rdata['Condition'] = rdata['Condition'].replace(condition_mapping)

    #     filepath = 'datapaper/'
    #     rdata = pd.read_csv(filepath+'meldataclean.csv')
    #     rdata = rdata.drop('Unnamed: 0',axis=1)
    #     rdata.rename(columns={'BioReplicate': 'Condition'}, inplace=True)

    #     rdata0 = pd.read_csv('egf/dataegf.csv')
    #     rdata1 = pd.read_csv('egf/dataegfintAKT0.csv')
    #     rdata0['condition'] = 0
    #     rdata1['condition'] = 1
    #     rdata = pd.concat([rdata0, rdata1], ignore_index=True)
    X_test = rdata.drop('Condition', axis=1)
    y_test = rdata['Condition']
    #     for _ in range(10):  # Run 10 times
    for _ in range(100):  # Randomly select samples 10 times
        # Randomly select 'sample_size' samples
        selected_samples = np.random.choice(len(data), size=sample_size, replace=replace)
        X_sampled = X.iloc[selected_samples]
        y_sampled = y.iloc[selected_samples]

        # Split the data into training and testing sets
        #             X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

        # Create a random forest classifier
        clf = RandomForestClassifier(random_state=42)

        # Train the classifier
        #             clf.fit(X_train, y_train)
        clf.fit(X_sampled, y_sampled)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate classification error
        error = 1 - accuracy

        accuracies.append(accuracy)
        errors.append(error)
        min_accuracies.append(np.min(accuracies))
        max_accuracies.append(np.max(accuracies))

    mina = np.min(accuracies)
    maxa = np.max(accuracies)

    avg_accuracy = np.mean(accuracies)
    avg_error = np.mean(errors)

    return avg_accuracy, mina, maxa
