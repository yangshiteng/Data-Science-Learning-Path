![image](https://user-images.githubusercontent.com/60442877/213952664-f475bf6c-c16b-4959-8ee1-2b143009ab0a.png)
![image](https://user-images.githubusercontent.com/60442877/213952684-26eb223d-6616-423a-8c8b-61ed2304f855.png)
![image](https://user-images.githubusercontent.com/60442877/213952730-0f1ed531-17fa-4fd7-9173-a23ef5dcb23d.png)

# Solution 1

    with temp1 as(
    select id, company, salary, row_number() over (partition by company order by salary,id) as emp_row_no
    from Employee),
    temp2 as(
    select company, count(salary) as salary_number from Employee group by company
    ),
    temp3 as(
        select company, if(salary_number % 2 != 0, (salary_number+1)/2, salary_number/2) as median_numb from temp2
        union
        select company, if(salary_number % 2 != 0, (salary_number+1)/2, salary_number/2+1) as median_numb from temp2
    )
    
    select id, company, salary
    from temp1
    where (company, emp_row_no) in (select * from temp3)

# Solution 2:

    select Id, Company, Salary
    from (select id, company, salary, 
           row_number() over(partition by company order by salary, id) salaryrank, 
           count(*) over(partition by company) tte 
        from employee) as foo
    where salaryrank >= tte/2 and salaryrank <= tte/2+1

![image](https://user-images.githubusercontent.com/60442877/214458582-d64ece44-a014-433f-b7f3-5c1af28d6358.png)
![image](https://user-images.githubusercontent.com/60442877/214458591-ca227946-8a7d-4165-9598-e250d4e0567c.png)
![image](https://user-images.githubusercontent.com/60442877/214458533-f4c090f2-13bb-43b7-ad30-1452546326cd.png)

    with t as (select *, sum(frequency) over(order by number) freq, (sum(frequency) over())/2 median_num
               from numbers)

    select avg(number) as median
    from t
    where median_num between (freq-frequency) and freq
