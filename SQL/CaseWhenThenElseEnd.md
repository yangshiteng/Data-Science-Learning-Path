![image](https://user-images.githubusercontent.com/60442877/207227474-2f96d56e-b2b2-4ea0-b1aa-b3c8a1c08c81.png)

![image](https://user-images.githubusercontent.com/60442877/207227556-fc4e526f-8408-48ff-b887-8b99ce9c4426.png)

![image](https://user-images.githubusercontent.com/60442877/207227575-9bf59d30-6363-42f1-8e1b-70f557f32c60.png)

![image](https://user-images.githubusercontent.com/60442877/212991247-791368c8-6b4f-46d3-b2ca-93a48b4e372b.png)

# LeetCode Question

![image](https://user-images.githubusercontent.com/60442877/217423384-3ce992f8-ed74-483d-8c4e-6c9c95082b19.png)
![image](https://user-images.githubusercontent.com/60442877/217423450-224c4211-bd3d-4342-be0b-33dadfbd755f.png)

    select id, 
    case 
        when p_id is null then 'Root'
        when (p_id is not null) and (id not in (select p_id from Tree where p_id is not null)) then 'Leaf'
        else 'Inner'
    end as type
    from Tree

![image](https://user-images.githubusercontent.com/60442877/217421742-e017f99a-e598-4315-b404-8b1220dafc86.png)
![image](https://user-images.githubusercontent.com/60442877/217421883-4115263d-213d-4445-aa8a-96b32f0c50ba.png)
