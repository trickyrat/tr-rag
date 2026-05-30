# Sqlalchemy Cheatsheet

## Basic query and where clauses

```python
from sqlalchemy import select, and_, or_

# single condition
stmt = select(User).where(User.name == "Alice")

# multiple conditions (AND)
stmt = select(User).where(User.name == "Alice", User.age > 18)

# AND combination
stmt = select(User).where(and_(User.name == "Alice", User.age > 18))

# OR combination
stmt = select(User).where(or_(User.name == "Alice", User.name == "Bob"))

# use & in condition
# note: & has higher priority than ==, each condition must add parentheses
stmt = select(User).where((User.name == "Alice") & (User.age > 18))

# Execute
result = session.execute(stmt)
users = result.scalars().all()   # return List of Model
```

## IN clause

```python
# column.in_([values])
stmt = select(User).where(User.name.in_(["Alice", "Bob", "Charlie"]))

# subquery with IN
subq = select(Address.user_id).where(Address.city == "New York")
stmt = select(User).where(User.id.in_(subq))
```

## JOIN Query

join() will generate inner join, outerjoin() will generate left outer join.

```python
# implicit join based on foreign key
stmt = select(User.name, Address.email).join(Address)

# explicit join based on ON condition
stmt = select(User.name, Address.email).join(Address, User.id == Address.user_id)

# letf outer join
from sqlalchemy import outerjoin
stmt = select(User.name, Address.email).outerjoin(Address)

# chain join
stmt = select(...).join(Address).join(SomeOther)

# get result (multiple columns)
for row in session.execute(stmt):
    print(row.name, row.email)
```

## Case When Statement

```python
from sqlalchemy import case

stmt = select(
    User.name,
    case(
        (User.age < 18, "Teenager"),
        (User.age < 60, "Adult"),
        else_="Elderly"
    ).label("age_group")
)
# SQL：CASE WHEN ... THEN ... ELSE ... END AS age_group
```

## Sort and Pagination

```python
# 升序
stmt = select(User).order_by(User.age.asc())

# 降序，多级排序
stmt = select(User).order_by(User.age.desc(), User.name.asc())

# 分页 (LIMIT / OFFSET)
stmt = select(User).limit(10).offset(20)   # 跳过 20 条，取 10 条
```

## Aggregate and group Functions

```python
from sqlalchemy import func

# 简单聚合
stmt = select(func.count(User.id))

# 分组 + HAVING
stmt = select(
    Address.city,
    func.count(User.id).label("cnt")
).join(User.addresses).group_by(Address.city).having(func.count(User.id) > 5)
```

## 子查询与 CTE

### 子查询：

```python
subq = select(Address.user_id).where(Address.city == "London").subquery()
stmt = select(User).where(User.id.in_(select(subq.c.user_id)))
# 注意：子查询列必须用 .c.column 引用
```
### CTE（公用表表达式）：

```python
cte = select(Address.user_id, func.count().label("addr_cnt")).group_by(Address.user_id).cte()
stmt = select(User.name, cte.c.addr_cnt).join(cte, User.id == cte.c.user_id)
```

##  插入、更新、删除 （2.0 风格）

```python
# 插入
from sqlalchemy import insert
stmt = insert(User).values(name="Tom", age=25)
session.execute(stmt)

# 批量插入
stmt = insert(User).values([{"name": "A"}, {"name": "B"}])
session.execute(stmt)

# 更新
from sqlalchemy import update
stmt = update(User).where(User.name == "Tom").values(age=30)
session.execute(stmt)

# 删除
from sqlalchemy import delete
stmt = delete(User).where(User.name == "Tom")
session.execute(stmt)

session.commit()
```