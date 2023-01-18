![image](https://user-images.githubusercontent.com/60442877/206073694-61f8ae1e-6b8c-4536-8693-f6a614bd9a5b.png)

![image](https://user-images.githubusercontent.com/60442877/206073780-dbf56e31-da13-40e1-8065-5f2a2a2b322b.png)

![image](https://user-images.githubusercontent.com/60442877/206074067-a3119724-d09d-4f02-a1a1-c68b145148e3.png)

# Leetcode Question

![image](https://user-images.githubusercontent.com/60442877/213220493-c05564d2-ec00-482e-8f9e-43164d763e27.png)
![image](https://user-images.githubusercontent.com/60442877/213220556-247d5d64-93fd-4a18-8fa6-4439ad9c356e.png)

    select distinct num as ConsecutiveNums
    from Logs
    where (id+1, num) in (select * from Logs) and (id+2, num) in (select * from Logs)

    select distinct l1.num as ConsecutiveNums
    from Logs l1, Logs l2, Logs l3
    where l1.num = l2.num and l2.num = l3.num and l1.id+1 = l2.id and l2.id+1 = l3.id


![image](https://user-images.githubusercontent.com/60442877/213299179-385e2537-1fdc-4c10-a66b-75b3e61c167f.png)
![image](https://user-images.githubusercontent.com/60442877/213299265-58c53bc8-3b41-4920-9538-f535e6874f48.png)

    SELECT
    Department.name AS 'Department',
    Employee.name AS 'Employee',
    Salary
    FROM
    Employee
        JOIN
    Department ON Employee.DepartmentId = Department.Id
    WHERE
    (Employee.DepartmentId , Salary) IN
    (   SELECT
            DepartmentId, MAX(Salary)
        FROM
            Employee
        GROUP BY DepartmentId
    )
    ;

    select D.name as Department, P1.name as Employee, P1.salary as Salary
    from (select name, salary, departmentId, dense_rank() over(partition by departmentId order by salary DESC) AS emp_dense_rank from Employee) P1 join Department D on P1.departmentId = D.id
    where P1.emp_dense_rank = 1

    
    
