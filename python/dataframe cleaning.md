# 1. drop()

    DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')

![image](https://user-images.githubusercontent.com/60442877/231341535-ae8e49db-1cca-4091-a95c-9dc291728c39.png)

![image](https://user-images.githubusercontent.com/60442877/231340912-fe392756-4584-4768-a5ee-896a7093d244.png)

![image](https://user-images.githubusercontent.com/60442877/231340938-b8728e1f-be1a-4859-b7c1-b7283fe34c8d.png)


# 2. dropna()

    DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

![image](https://user-images.githubusercontent.com/60442877/231342395-f774e8d2-76b4-43b8-96e4-51e432a462cc.png)

![image](https://user-images.githubusercontent.com/60442877/232176136-b085cc31-d534-4075-9cd8-14aff8935c55.png)

![image](https://user-images.githubusercontent.com/60442877/232176141-c3bc2a00-ac9b-4787-8c88-8d0d901b580f.png)


# 3. fillna()

    DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)

![image](https://user-images.githubusercontent.com/60442877/232175661-c3d97fe0-20a2-48a6-aca5-6ddd521d51f4.png)

![image](https://user-images.githubusercontent.com/60442877/232175738-3350a18c-03b6-4abf-8f79-34892832a603.png)

![image](https://user-images.githubusercontent.com/60442877/232175712-47f26892-e24c-4405-97d3-0efbfed96efe.png)


# 4. replace()

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

# 9. merge()

![image](https://user-images.githubusercontent.com/60442877/232178546-2d5e2434-6308-431f-9200-04054c394cee.png)

    pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)

![image](https://user-images.githubusercontent.com/60442877/232178650-c27806e3-25ff-4552-a575-2e80cbfa2856.png)

![image](https://user-images.githubusercontent.com/60442877/232178686-bfcbdf1a-a879-4491-a993-ef27a6a3d957.png)

![image](https://user-images.githubusercontent.com/60442877/232178690-63d2ea7a-5200-4285-8caf-52cbe469fe03.png)

# 10. concat()

![image](https://user-images.githubusercontent.com/60442877/232178872-1b44cf25-a298-466f-92a9-34d6fa5f2a62.png)

    pd.concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True)

![image](https://user-images.githubusercontent.com/60442877/232178882-b574c28e-a45c-4f22-9dd6-9eec9cb76b36.png)

![image](https://user-images.githubusercontent.com/60442877/232179016-d64ea06f-90a7-4940-8ef1-211c7f46913c.png)

![image](https://user-images.githubusercontent.com/60442877/232179027-d1bcffe4-8cdb-470f-ac06-b35c98e38499.png)

![image](https://user-images.githubusercontent.com/60442877/232179034-9a53e86b-d727-48aa-8257-e3055c2a11da.png)


# 11. pivot_table()

![image](https://user-images.githubusercontent.com/60442877/232179232-5eab8a3a-ba97-4a4b-a596-dbd7d7d052cf.png)

    pd.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)

![image](https://user-images.githubusercontent.com/60442877/232179268-ca85a104-08f4-46be-8b62-a7a22a93633d.png)

![image](https://user-images.githubusercontent.com/60442877/232179479-02368ad8-d252-4b93-8eff-1c60df3b600b.png)

![image](https://user-images.githubusercontent.com/60442877/232179486-642a9854-dcf5-4606-b363-775e8d95ffc7.png)

# 12. melt()

![image](https://user-images.githubusercontent.com/60442877/232179567-05da3c43-8758-4072-ae68-4cf8fccdb7f1.png)

    pd.melt(frame, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None, ignore_index=True)

![image](https://user-images.githubusercontent.com/60442877/232179577-773d1886-5469-416e-b056-ef2acd666e42.png)

![image](https://user-images.githubusercontent.com/60442877/232179674-112bd19c-40cf-4997-9a9b-ff1461f4ea7d.png)

![image](https://user-images.githubusercontent.com/60442877/232179681-37d6b5df-22a7-4763-a0b2-5ed31927f1b1.png)





![image](https://user-images.githubusercontent.com/60442877/231333149-617e390b-615c-4225-b257-b385348b06b5.png)

