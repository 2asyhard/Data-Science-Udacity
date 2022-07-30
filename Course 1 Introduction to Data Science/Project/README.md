# Stackoverflow survey 2017 Data Analysis
[Related article](https://medium.com/@tetaeho/find-feature-that-best-predict-the-developers-salary-4598c4bdc62a)
***

## Motive
Based on Cross-Industry Standard Process of Data Mining or CRISP-DM, 
the Stackoverflow survey 2017 dataset were collected and investigated.

The business questions that I'm trying to solve:

1. How would you rank the most used programming languages among respondents?
2. Is there a difference in programming languages between people with high salaries and those with low salaries?
3. Which feature can best predict salary?
4. Can you find a pattern related to salary using the best column in the results of Q3?

***

## Libraries

* Pandas
* Numpy
* Scikit-learn

***

## Dataset

* Stackoverflow survey 2017
    * https://www.kaggle.com/datasets/stackoverflow/so-survey-2017

***

## Result Summary

1. The programming languages most frequently used by developers appeared in the order of Javascript, SQL, Java, and C#, and the least used programming languages appeared in the order of Hack, Julia, and Dart.
2. I thought there might be a difference in `Salary` depending on the language the developers use, but it turns out that there is almost no difference in the language used by the top 20% and the bottom 20%.
3. Linear regression was performed to find the column that predicted the salary best among columns in the dataset, and as a result, `Currency` could have the best r2 score of 0.44 or higher. It was judged that it was difficult to accurately calculate Salary by having a `HaveWorkedLanguage` value of 0.05 indicating the language being used.
4. In the results obtained through linear regression, a pattern was found using the column (`Currency`) with the best r2 score, and it was found that the top 20% received salary in USD.
