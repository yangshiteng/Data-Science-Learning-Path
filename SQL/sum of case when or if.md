![image](https://user-images.githubusercontent.com/60442877/213339869-f9097653-6b63-481f-bd07-584068845a7d.png)
![image](https://user-images.githubusercontent.com/60442877/213339896-e53217db-a5d9-49d4-9c27-20fdc6ed34b3.png)
![image](https://user-images.githubusercontent.com/60442877/213339909-28b6e400-9a3b-44bc-8645-e22f4296f6fe.png)

# Solution 1:

    select P1.request_at as Day, round(coalesce(P2.cancel_number,0)/P1.total_number,2) as 'Cancellation Rate'
    from 
    (select request_at, count(id) as total_number
    from Trips
    where client_id not in (select users_id from Users where banned = 'Yes') and driver_id not in (select users_id from Users where banned = 'Yes')
    group by request_at) as P1 
    left join 
    (select request_at, count(id) as cancel_number
    from Trips
    where client_id not in (select users_id from Users where banned = 'Yes') and driver_id not in (select users_id from Users where banned = 'Yes') and status != 'completed'
    group by request_at) as P2
    on P1.request_at = P2.request_at
    where P1.request_at between '2013-10-01' and '2013-10-03'
  
# Solution 2 (sum of case when then else end):

    select 
        request_at as "Day",
        round(
            (sum(case when status = "cancelled_by_driver" or status = "cancelled_by_client" then 1 else 0 end) / count(status)), 2
        ) as "Cancellation Rate"
    from
        Trips
    where 
        client_id not in (select users_id from Users where role = 'client' and banned ='Yes') 
    and 
        driver_id not in (select users_id from Users where role = 'driver' and banned ='Yes') 
    and 
        request_at >= "2013-10-01" and request_at <= "2013-10-03"
    group by 
        request_at

# Solution 3 (sum of if):

    select request_at as Day, round(sum(if(status = "cancelled_by_driver" or status = "cancelled_by_client", 1,0 )) / count(id) ,2) as "Cancellation Rate"
    from Trips 
    where client_id not in (select users_id from Users  where banned = "Yes") 
    and driver_id  not in (select users_id from Users  where banned = "Yes")
    and request_at between "2013-10-01" and "2013-10-03"
    group by request_at 




