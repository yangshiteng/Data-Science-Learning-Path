# Introduction of Ensemble Learning (集成学习) 

![image](https://user-images.githubusercontent.com/60442877/196843226-4712cbf4-9df1-49ba-b12f-67c4704a5d00.png)

![image](https://user-images.githubusercontent.com/60442877/196833314-b765b52c-2405-45ab-85d0-0dce9c3e024b.png)

![image](https://user-images.githubusercontent.com/60442877/196834070-81c9ab13-5301-4a46-92ef-d198615c3cee.png)

![image](https://user-images.githubusercontent.com/60442877/196834094-6dd12e0e-40d4-411b-a8a6-bfff9222125b.png)

![image](https://user-images.githubusercontent.com/60442877/235266591-a14ed475-705f-433f-a59e-096bbd2968bb.png)

![image](https://user-images.githubusercontent.com/60442877/235268916-bf380d68-e6d0-4d3e-8a51-4f14f6b23a61.png)


# Bagging 装袋算法 (Bootstrap Aggregating 引导聚集算法) 

![image](https://user-images.githubusercontent.com/60442877/196834603-05a48da7-e218-4655-aeeb-a10e05c34cac.png)

![image](https://user-images.githubusercontent.com/60442877/196834911-7d5374b9-273a-4af1-96b4-09283f61fd9b.png)

![image](https://user-images.githubusercontent.com/60442877/196835210-cb8b47ec-fa93-405c-9b2a-b0bfee692cd2.png)

![image](https://user-images.githubusercontent.com/60442877/196836148-c7473371-1c3d-49ac-929e-7eca25b30985.png)

![image](https://user-images.githubusercontent.com/60442877/235270096-0e0a0d22-bc3f-4037-aab1-a5322cadec47.png)


# Boosting (提升算法)

## Tutorial 1

![image](https://user-images.githubusercontent.com/60442877/235272674-7f475e41-d1f1-4504-9321-9127fec70d74.png)

![image](https://user-images.githubusercontent.com/60442877/196840284-1da5dc83-95f4-4171-a648-448cb56336aa.png)

![image](https://user-images.githubusercontent.com/60442877/196841083-68e9f19f-d145-4175-8f1e-f0d10905bf35.png)

![image](https://user-images.githubusercontent.com/60442877/196842709-c6843629-bf54-4ad5-8cee-1d5c581c7154.png)

![image](https://user-images.githubusercontent.com/60442877/196842744-23cac8da-ca10-431f-b960-5cc449c40387.png)

![image](https://user-images.githubusercontent.com/60442877/196842774-e943f157-ba85-433f-b117-5f20a701dfa7.png)

![image](https://user-images.githubusercontent.com/60442877/235273037-fd01ab59-8b84-4598-a268-542e2062a8f7.png)

## Tutorial 2

![image](https://user-images.githubusercontent.com/60442877/235279795-8b963ec8-f190-416e-8a1d-87388359835f.png)
* 基础学习器所造成的误差可以通过很多不同的方法被下一个基础学习器纠正，不同的提升算法的核心差异就在于使用了不同的误差纠正方法

![image](https://user-images.githubusercontent.com/60442877/235280042-8d71fa4a-876b-4c86-b948-0e5dc42865af.png)

### Adaboost (Adaptive Boosting) (自适应提升算法)

![image](https://user-images.githubusercontent.com/60442877/235280895-84b4514f-cc1e-417c-9737-4a93839765fa.png)

![image](https://user-images.githubusercontent.com/60442877/235282681-e55a7fe6-0702-4df6-873b-f0202b5d773d.png)

    # For this basic implementation, we only need these modules
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier

    # Load the well-known Breast Cancer dataset
    # Split into train and test sets
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=23)

    # The base learner will be a decision tree with depth = 2
    tree = DecisionTreeClassifier(max_depth=2, random_state=23)

    # AdaBoost initialization
    # It's defined the decision tree as the base learner
    # The number of estimators will be 5
    # The penalizer for the weights of each estimator is 0.1
    adaboost = AdaBoostClassifier(base_estimator=tree, n_estimators=5, learning_rate=0.1, random_state=23)

    # Train!
    adaboost.fit(x_train, y_train)

    # Evaluation
    print(f"Train score: {adaboost.score(x_train, y_train)}")
    print(f"Test score: {adaboost.score(x_test, y_test)}")

![image](https://user-images.githubusercontent.com/60442877/235282730-603b275b-d9ee-4fb1-9075-d1e7776dc363.png)

### Gradient Boosting (梯度上升算法)

![image](https://user-images.githubusercontent.com/60442877/235282828-b4b72ce6-a27c-437b-a297-d14922cf9a84.png)

![image](https://user-images.githubusercontent.com/60442877/235283064-eb636c35-f0cc-467c-a050-b80a995e8f81.png)

    # For this basic implementation, we only need these modules
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier

    # Load the well-known Breast Cancer dataset
    # Split into train and test sets
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=23)

    # Gradient Boosting initialization
    # The base learner is a decision tree as default
    # The number of estimators is 5
    # The depth for each deciion tree is 2
    # The learning rate for each estimator in the sequence is 1
    gradientBoosting = GradientBoostingClassifier(n_estimators=5, learning_rate=1, max_depth=2, random_state=23)

    # Train!
    gradientBoosting.fit(x_train, y_train)

    # Evaluation
    print(f"Train score: {gradientBoosting.score(x_train, y_train)}")
    print(f"Test score: {gradientBoosting.score(x_test, y_test)}")
    
![image](https://user-images.githubusercontent.com/60442877/235283107-fbcdd8ea-6957-4b9a-8296-e5b603b54d0d.png)


# Stacking (堆叠算法)

![image](https://user-images.githubusercontent.com/60442877/196837688-119e3a14-ed25-4faf-8240-8b016b23df48.png)

![image](https://user-images.githubusercontent.com/60442877/196837902-4e907651-05cd-4484-84b6-26bc9014aea1.png)

![image](https://user-images.githubusercontent.com/60442877/196837927-77cb9b6c-260b-4e9e-a5f9-08ebcd8e4c86.png)

![image](https://user-images.githubusercontent.com/60442877/235271088-8f3bcc5e-2769-43bc-883a-7939fde5eb1b.png)

![image](https://user-images.githubusercontent.com/60442877/235278456-fe060934-6089-4c4d-94b2-588bea18e3e8.png)


# Bagging vs Boosting vs Stacking

![image](https://user-images.githubusercontent.com/60442877/235278094-1ebd638b-2748-4a40-98b0-093d7a74721a.png)

![image](https://user-images.githubusercontent.com/60442877/235278404-30f8cb3f-1f17-499e-8af4-bd60c7baa96d.png)

![image](https://user-images.githubusercontent.com/60442877/235278420-df309a9d-6a88-41bd-a633-aaa571a415bf.png)



