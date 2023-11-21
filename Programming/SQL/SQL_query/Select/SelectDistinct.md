![image](https://user-images.githubusercontent.com/60442877/205424450-9fa1dbe9-3377-4933-8e71-119f60ee0424.png)

# Leetcode Question (not just applied for single column, but also duplicate row)

![image](https://user-images.githubusercontent.com/60442877/218227852-e9edb1b8-64af-4452-b8be-2e7542245b55.png)
![image](https://user-images.githubusercontent.com/60442877/218227860-8b1d6c4c-199e-4061-9c74-29a91f981284.png)

    with temp1 as(
    select E.department_id, S.amount, Date_Format(S.pay_date, "%Y-%m") as pay_month
    from Salary S left join Employee E on S.employee_id = E.employee_id),
    temp2 as(
    select department_id, pay_month, Avg(amount) over(partition by pay_month, department_id) as department_average_salary, Avg(amount) over(partition by pay_month) as company_average_salary
    from temp1)

    select distinct pay_month, department_id, 
    case
        when company_average_salary = department_average_salary then 'same'
        when company_average_salary > department_average_salary then 'lower'
        else 'higher'
    end as comparison
    from temp2

