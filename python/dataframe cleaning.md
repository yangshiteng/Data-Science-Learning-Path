# 1. drop()

![image](https://user-images.githubusercontent.com/60442877/232180566-29291716-905d-4ad4-aa5e-6f29523e9489.png)

    DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')

![image](https://user-images.githubusercontent.com/60442877/231341535-ae8e49db-1cca-4091-a95c-9dc291728c39.png)

![image](https://user-images.githubusercontent.com/60442877/231340912-fe392756-4584-4768-a5ee-896a7093d244.png)

![image](https://user-images.githubusercontent.com/60442877/231340938-b8728e1f-be1a-4859-b7c1-b7283fe34c8d.png)


# 2. dropna()

![image](https://user-images.githubusercontent.com/60442877/232180547-24d2caf6-a1d6-4f5b-b7ce-61585301a291.png)

    DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

![image](https://user-images.githubusercontent.com/60442877/231342395-f774e8d2-76b4-43b8-96e4-51e432a462cc.png)

![image](https://user-images.githubusercontent.com/60442877/232176136-b085cc31-d534-4075-9cd8-14aff8935c55.png)

![image](https://user-images.githubusercontent.com/60442877/232176141-c3bc2a00-ac9b-4787-8c88-8d0d901b580f.png)


# 3. fillna()

![image](https://user-images.githubusercontent.com/60442877/232180556-11920a4a-aebb-4977-b01f-6911ab7e70cb.png)

    DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)

![image](https://user-images.githubusercontent.com/60442877/232175661-c3d97fe0-20a2-48a6-aca5-6ddd521d51f4.png)

![image](https://user-images.githubusercontent.com/60442877/232175738-3350a18c-03b6-4abf-8f79-34892832a603.png)

![image](https://user-images.githubusercontent.com/60442877/232175712-47f26892-e24c-4405-97d3-0efbfed96efe.png)


# 4. replace()

![image](https://user-images.githubusercontent.com/60442877/232180537-0be487dc-4751-4d58-b2c1-937a06cf3171.png)

    DataFrame.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')

![image](https://user-images.githubusercontent.com/60442877/232176264-2189d503-b864-41dc-acd2-505d8a79eef5.png)

![image](https://user-images.githubusercontent.com/60442877/232176758-d646d3fc-f10e-4ae8-ae0a-2118074b5066.png)

![image](https://user-images.githubusercontent.com/60442877/232176765-bf596cf6-7d0f-4ae0-a4d0-c9048e2f1866.png)

# 5. rename()

![image](https://user-images.githubusercontent.com/60442877/232176994-d9cd036b-6f46-474d-a20d-85094d396982.png)

    DataFrame.rename(mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='raise')

![image](https://user-images.githubusercontent.com/60442877/232177091-1b15b9de-e363-4900-913d-b8de5708b2ca.png)

![image](https://user-images.githubusercontent.com/60442877/232177171-bb6eedcf-0ef8-4bee-b595-019cbee06ae1.png)

![image](https://user-images.githubusercontent.com/60442877/232177178-ab1cc955-3c59-41b8-8835-3958a643e0c5.png)

# 6. astype()

![image](https://user-images.githubusercontent.com/60442877/232177473-06feb0fc-6337-4094-9c4a-3722bf6afa09.png)

    DataFrame.astype(dtype, copy=True, errors='raise')

![image](https://user-images.githubusercontent.com/60442877/232177485-aa2adbd1-3735-4152-a36e-79f9883563b1.png)

![image](https://user-images.githubusercontent.com/60442877/232177505-8f2d291e-6383-41c9-8350-11ec38b3b5f8.png)

![image](https://user-images.githubusercontent.com/60442877/232177507-d717c726-fc85-4f14-a0c4-35d0874061fd.png)

# 7. apply()

![image](https://user-images.githubusercontent.com/60442877/232177636-a3554cc1-d8c7-4843-851b-dab6ff758504.png)

    DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwargs)

![image](https://user-images.githubusercontent.com/60442877/232177698-709ef0ef-0c12-4c7a-83c0-e2227ca5e075.png)

![image](https://user-images.githubusercontent.com/60442877/232177782-88efd57e-c5be-42cc-9681-dad769928f3c.png)

![image](https://user-images.githubusercontent.com/60442877/232177787-5263e59d-9fb3-4b08-8df8-1ff5268139b7.png)

# 8. applymap()

![image](https://user-images.githubusercontent.com/60442877/232178250-8d178222-5cfe-487d-9cff-d6fcbeeb6e77.png)

    DataFrame.applymap(func)

![image](https://user-images.githubusercontent.com/60442877/232178272-eeec58ed-218b-4df4-965f-047ac310ae0b.png)

![image](https://user-images.githubusercontent.com/60442877/232178284-a43ca021-906f-438a-98f8-983db2433d45.png)


# 9. duplicated()

![image](https://user-images.githubusercontent.com/60442877/232180424-c64759e8-6604-4bb7-b2b3-73d7a6d3608a.png)

    DataFrame.duplicated(subset=None, keep='first')

![image](https://user-images.githubusercontent.com/60442877/232180433-14411ef3-3ccf-40ac-8e67-120e5b0735c9.png)

![image](https://user-images.githubusercontent.com/60442877/232180489-fc635c80-0edf-484a-b43b-69ffe1077df9.png)

![image](https://user-images.githubusercontent.com/60442877/232180498-634dde88-2d81-4f8d-8f13-37add371e2c5.png)

# 10. drop_duplicates()

![image](https://user-images.githubusercontent.com/60442877/232265742-86075abe-1dff-463f-9c30-0fb2e4e759a7.png)

    DataFrame.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)

![image](https://user-images.githubusercontent.com/60442877/232265764-26644021-3785-463f-bd1a-7ac791dfa60e.png)

![image](https://user-images.githubusercontent.com/60442877/232265783-39253165-8f21-4bc5-83f4-e14d261302f7.png)

![image](https://user-images.githubusercontent.com/60442877/232265788-9f397de7-983b-4f01-a0f6-097670d66e0d.png)




