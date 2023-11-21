# SQL Common Knowledge

* [Introduction](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_common_knowledge/Introduction.md)
* [Alias](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_common_knowledge/Alias.md)
* [SQL Stored Procedures for SQL Server](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_common_knowledge/SQL_Stored_Procedures_for_SQL_Server.md)
* [SQL Comments](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_common_knowledge/SQL_comments.md)
* [With Clause (defining a temporary table)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_common_knowledge/with_clause.md)

# SQL Query

## Select (which column)

* [Select \*](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/SelectALL.md)
* [Select Distinct](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/SelectDistinct.md)
* [Select Top](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/SelectTop.md)
* [Min() and Max()](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/minandmax.md)
* [Count, Avg and Sum](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/Count_Avg_Sum.md)
* [Column concatenate](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/ColumnCombine.md)
* [Null Functions (IFNULL, ISNULL, COALESCE, etc...) ](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/NullFunctions.md)
* [SQL Arithmetic Operators](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/SQL_Arithmetic_Operators.md)
* [Cast() or Convert() converts a value of any type into the specified datatype](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/CastFunction.md)
* [If() returns a value if a condition is TRUE, or another value if a condition is FALSE](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/If_function.md)
* [Sum() of case_when_then_else_end or Sum() of if()](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/sum_of_case_when_or_if.md)
* [Group Concatenation (Concatenate Multiple rows in the same group into a single field)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/Multiperow_to_one_row.md)
* [Window Function, Rank(), Dense_Rank(), Row_Number()](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Select/Window_Function.md)

## 2. From (which table) 

* [Inner Join, Left Join, Right Join, Full Join](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/From/SQL_join.md)
* [Self Join, Cross Join](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/From/self_join.md)

## 3. Where (which rows)
### (can not be used with aggregate function)
### (together with From, all the logics are applied to the table defined in from...where...)

* [Comparison Operators](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Where/Comparison_Operators.md)
* [And, Or, Not](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Where/And_Or_Not.md)
* [Is Null, Is Not Null](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Where/Nullvalue.md)
* [Like](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Where/likeoperator.md)
* [In, Not In](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Where/In_operator.md)
* [Between, Not Between](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Where/Between.md)
* [Exists (test for the existence of any record in a subquery)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Where/Exists.md)
* [Any](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Where/Any.md)
* [All](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Where/ALL.md)

## 4. Group by (which column)

* By using Group By clause, the table is divided into groups. And in Select Clause, some computations are applied to each group to generate a single value. So, one group one value.
* [Group by clause](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Groupby/Groupby.md)

## 5. Having (which condition with which aggregate function)

* [Having clause](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Having/Having.md)

## 6. Order By (which column)

* [Order By clause](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Orderby/Order_by.md)
* [Order by Case When Then Else End](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Orderby/Order_by_Case_When_Then_Else_End.md)
* [Order by Aggregate functions: Count, Max, Min, Sum, Avg](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Orderby/OrderbyAggregatefunctionsCountMaxMinSumAvg.md)
* [Order by multiple columns](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Orderby/Orderbymultilecolumns.md)

## 7. limit & offset

* [limit & offset (MySQL)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/limit_offset/limit_and_offset.md)

## 8. Union

* [Union, Union All](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/Union/Union.md)

## 9. Conditional Statement

* [Case When Then Else End](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/SQL_query/conditional_statement/CaseWhenThenElseEnd.md)

# Table Edit

## 1. Insert new records into a table
* [Insert Into (which table (which column) ) Values (which value)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/Insert_into.md)

## 2. Modify the existing records in a table
* [Update (which table) Set (which column = which value) Where (which rows)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/update.md)

## 3. Delete existing records in a table
* [Delete From (which table) Where (which rows)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/Delete.md)

## 4. Copy data from one table into a new table
* [Select (which column) Into (which new table (In which database) ) From (which old table) Where (which rows)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/SelectInto.md)

## 5. Copy data from one table and insert it into another table
* [Insert Into (which target table (which column) ) Select (which column) From (which source table) Where (which rows)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/InsertIntoSelect.md)

## 6. Create, Drop, Alter Table
* [Create Table](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/CreateTable.md)
* [Drop Table](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/Droptable.md)
* [Alter Table](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/Altertable.md)

## 7. Create, Drop Index
* [Create Index](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/CreateIndex.md)
* [Drop Index](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/Dropindex.md)

## 8. Constraint
* [Constraints](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/Constraints.md)

## 9. View (Virtual Table)
* [Views (Virtual Table)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/Views.md)

## 10. SQL Data Types
* [SQL Data Types for MySQL, SQL Server, and MS Access](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/table_edit/SQLDataTypes.md)

# DataBase Management

## [1. Create, Drop, Backup Database](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/Database_management/CreateDB.md)
## [2. SQL Injection](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/Database_management/SQL_Injection.md)
## [3. SQL Hosting](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/Database_management/SQL_hosting.md)

# Functions (MySQL)

## [1. String Functions](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/Functions/StringFunctions.md)
## [2. Numeric Functions](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/Functions/NumericFunctions.md)
## [3. Date Functions](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/Functions/Date_Functions.md)
## [4. Advanced Functions](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/Functions/Advanced_function.md)
## [5. Calculate Median (MySQL)](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Programming/SQL/Functions/median.md)

