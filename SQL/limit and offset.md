![image](https://user-images.githubusercontent.com/60442877/213044752-9b579884-33ab-411c-b490-ab247775cd47.png)

# Leetcode Question

![image](https://user-images.githubusercontent.com/60442877/213085447-2a885ccd-09f6-47bf-a687-011bb825967b.png)
![image](https://user-images.githubusercontent.com/60442877/213085463-35728f38-1624-4227-a848-125a6a46e0ea.png)

    CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
    BEGIN
    declare M int;
    SET M = N - 1;
      RETURN (
          # Write your MySQL query statement below.
        select coalesce((select distinct salary
          from Employee
          order by salary Desc
          limit M,1), null) 
      );
    END
