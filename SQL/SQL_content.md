# SQL Common Knowledge
* [1. Introduction](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Introduction.md)
* [2. Alias](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Alias.md)
* [3. SQL Stored Procedures for SQL Server](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/SQL%20Stored%20Procedures%20for%20SQL%20Server.md)
* [4. SQL Comments](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/SQL%20comments.md)
* [5. With Clause (defining a temporary table)](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/with%20clause.md)

# SQL Query

## 1. Select (which column)
  * [Select \*](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/SelectALL.md)
  * [Select Distinct](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/SelectDistinct.md)
  * [Select Top](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/SelectTop.md)
  * [Min() and Max()](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/minandmax.md)
  * [Count, Avg and Sum](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Count%2C%20Avg%20and%20Sum.md)
  * [Column concatenate](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/ColumnCombine.md)
  * [Null Functions (IFNULL, ISNULL, COALESCE, etc...) ](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/NullFunctions.md)
  * [SQL Arithmetic Operators](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/SQL%20Arithmetic%20Operators.md)
  * [Cast() or Convert() converts a value of any type into the specified datatype](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/CastFunction.md)
  * [If() returns a value if a condition is TRUE, or another value if a condition is FALSE](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/If_function.md)
  * [Window Function, Rank(), Dense_Rank(), Row_Number()](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Window%20Function.md)
  * [Sum() of case_when_then_else_end or Sum() of if()](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/sum%20of%20case%20when%20or%20if.md)
## 2. From (which table) 
  * [Inner Join, Left Join, Right Join, Full Join](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/Notes/SQL_join.md)
  * [Self Join, Cross Join](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/self%20join.md)
## 3. Where (which rows) (can not be used with aggregate function)
  * [Comparison Operators](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Comparison%20Operators.md)
  * [And, Or, Not](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/And%2C%20Or%2C%20Not.md)
  * [Is Null, Is Not Null](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Nullvalue.md)
  * [Like](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/likeoperator.md)
  * [In, Not In](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/In_operator.md)
  * [Between, Not Between](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Between.md)
  * [Exists (test for the existence of any record in a subquery)](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Exists.md)
  * [Any](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Any.md)
  * [All](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/ALL.md)
## 4. Group by (which column)
  * [Group by clause](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Groupby.md)
## 5. Having (which condition with which aggregate function)
  * [Having clause](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Having.md)
## 6. Order By (which column)
  * [Order By clause](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Order_by.md)
  * [Order by Case When Then Else End](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Order%20by%20Case%20When%20Then%20Else%20End.md)
  * [Order by Aggregate functions: Count, Max, Min, Sum, Avg](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Order%20by%20Aggregate%20functions:%20Count%2C%20Max%2C%20Min%2C%20Sum%2C%20Avg.md)
## 7. limit & offset
  * [limit & offset (MySQL)](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/limit%20and%20offset.md)
## 8. Union
  * [Union, Union All](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Union.md)
## 9. Multiple Clause Application 
  * [Case When Then Else End](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/CaseWhenThenElseEnd.md)


# Table Edit

## 1. Insert new records into a table
  * [Insert Into (which table (which column) ) Values (which value)](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Insert_into.md)
## 2. Modify the existing records in a table
  * [Update (which table) Set (which column = which value) Where (which rows)](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/update.md)
## 3. Delete existing records in a table
  * [Delete From (which table) Where (which rows)](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Delete.md)
## 4. Copy data from one table into a new table
  * [Select (which column) Into (which new table (In which database) ) From (which old table) Where (which rows)](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/SelectInto.md)
## 5. Copy data from one table and insert it into another table
  * [Insert Into (which target table (which column) ) Select (which column) From (which source table) Where (which rows)](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/InsertIntoSelect.md)
## 6. Create, Drop, Alter Table
  * [Create Table](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/CreateTable.md)
  * [Drop Table](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Droptable.md)
  * [Alter Table](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Alter%20table.md)
## 7. Create, Drop Index
  * [Create Index](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Create%20Index.md)
  * [Drop Index](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Dropindex.md)
## 8. Constraint
  * [Constraints](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Constraints.md)
## 9. View (Virtual Table)
  * [Views (Virtual Table)](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Views.md)
## 10. SQL Data Types
  * [SQL Data Types for MySQL, SQL Server, and MS Access](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/SQL%20Data%20Types%20for%20MySQL%2C%20SQL%20Server%2C%20and%20MS%20Access.md)

# DataBase Management

## [1. Create, Drop, Backup Database](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/CreateDB.md)
## [2. SQL Injection](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/SQL%20Injection.md)
## [3. SQL Hosting](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/SQL%20hosting.md)

# Functions (MySQL)

## [1. String Functions](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/StringFunctions.md)
## [2. Numeric Functions](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/NumericFunctions.md)
## [3. Date Functions](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Date%20Functions.md)
## [4. Advanced Functions](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/SQL/Advanced%20function.md)


