create database Stroke;
use Stroke;
select * from modified_stroke_data  limit 10;

select count(*) as stroke_cases
from modified_stroke_data
where stroke=1;

select gender,count(*) as stroke_count
from modified_stroke_data
where stroke=1
group by gender;

select work_type , avg(bmi) as avg_bmi
from modified_stroke_data
group by work_type; 

select smoking_status ,avg(avg_glucose_level) as avg_glucose
from modified_stroke_data
group by smoking_status;

select round((sum(case when stroke=1 then 1 else 0 end)/count(*))*100,2) as stroke_percentage
from modified_stroke_data
where hypertension=1;

select age_group,round(avg(stroke)*100,2) as stroke_rate_percentage
from modified_stroke_data
group by age_group;

select bmi_category,count(*) as stroke_count
from modified_stroke_data
where stroke=1
group by bmi_category;

select work_type , count(*) as stroke_count
from modified_stroke_data
where stroke=1
group by work_type 
order by stroke_count desc;

select residence_type,avg(age) as avg_age
from modified_stroke_data
where stroke=1
group by residence_type; 

create view  stroke_by_gender as
select gender,count(*) as stroke_count
from  modified_stroke_data
where stroke=1
group by gender;

create view avg_bmi_by_worktype as
select work_type,avg(bmi) as avg_bmi
from modified_stroke_data
group by work_type;

create view stroke_rate_by_agegroup as
select age_group,round(avg(stroke)*100,2) as stroke_rate_percentage
from modified_stroke_data
group by age_group;

create view hypertension_stroke_pct as
select round((sum(case when stroke=1 then 1 else 0 end)/count(*))*100,2) as stroke_pct
from modified_stroke_data
where hypertension=1;

create view stroke_by_bmi_category as 
select bmi_category, count(*) as stroke_count
from modified_stroke_data
where stroke=1
group by bmi_category;