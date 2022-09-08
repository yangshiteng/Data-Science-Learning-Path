![image](https://user-images.githubusercontent.com/60442877/187278753-7f860ea6-13d3-4a22-b5e0-f0993c27ebb0.png)

![image](https://user-images.githubusercontent.com/60442877/187279306-21da7e51-6174-4d61-bdcd-0a8cafa74c4c.png)

![image](https://user-images.githubusercontent.com/60442877/187280911-fe41e895-8cec-49c0-bb1d-100ea8342929.png)

![image](https://user-images.githubusercontent.com/60442877/187280988-99d6ce54-9e4e-452e-9050-cb7065b560a0.png)

![image](https://user-images.githubusercontent.com/60442877/187281124-56d6fe51-1085-462f-8e45-64563984221f.png)

![image](https://user-images.githubusercontent.com/60442877/187281524-4a158d6f-e72a-467c-8486-3e337d7ac809.png)

![image](https://user-images.githubusercontent.com/60442877/187281806-bc0615e2-986e-402e-a129-8d757306b295.png)

![image](https://user-images.githubusercontent.com/60442877/187282954-a1d68dda-842e-486b-a893-7707892f9d9c.png)

![image](https://user-images.githubusercontent.com/60442877/187283188-8e7b8b8e-7f9d-48f0-9c86-fbda0c13e975.png)

![image](https://user-images.githubusercontent.com/60442877/187283437-36a23144-6172-4524-aa43-3e1ab057d242.png)

![image](https://user-images.githubusercontent.com/60442877/187283836-eeb06630-9017-449e-9745-f2b87fda76f4.png)

![image](https://user-images.githubusercontent.com/60442877/187283917-d9d8a14f-be01-49f2-907b-5422da5f5ccc.png)

![image](https://user-images.githubusercontent.com/60442877/187284811-fba66240-c5c5-486b-bfc9-944e201376a5.png)

![image](https://user-images.githubusercontent.com/60442877/187285265-b2bbb1d6-33cd-441d-914d-c2fc9ffa23c0.png)

![image](https://user-images.githubusercontent.com/60442877/187285502-4518a49a-7a3f-4948-9000-d32254a9a0cb.png)

# Naive Bayes, How to make the classification?

* P(Spam Email | Data) is prop to P(Spam Email) * P(Data | Spam Email)
* P(Non-spam Email | Data) is prop to P(Non-Spam Email) * P(Data | Non-Spam Email)


1. Calculating P(Spam Email) * P(Data | Spam Email)
2. Calculating P(Non-Spam Email) * P(Data | Non-Spam Email)
3. Calculating the ratio P(Spam Email) * P(Data | Spam Email) over the sum of P(Spam Email) * P(Data | Spam Email) and P(Non-Spam Email) * P(Data | Non-Spam Email)
4. P(Spam Email | Data) is equal to the ratio calculated in step 3
5. Select an approriate threshold, for example 0.5, to make the classification

* Note: By Law of Total Probability, P(Data) is equal to the sum of P(Spam Email) * P(Data | Spam Email) and P(Non-Spam Email) * P(Data | Non-Spam Email)

![image](https://user-images.githubusercontent.com/60442877/189025687-529deba4-ac54-496b-8ea9-69e8ad6a5bf2.png)


