from neo4j import GraphDatabase
import re
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = 'companypaper2'

def find_missing_episodic_numbers(uri, user, password, database_name):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    missing_numbers = []
    try:
        with driver.session(database=database_name) as session:
            # 查询获取所有 Episodic 节点的 name 属性
            result = session.run("MATCH (n:Episodic) RETURN n.name")
            names = [record["n.name"] for record in result]

            if not names:
                return missing_numbers

            # 提取每个 name 最后一个 '_' 后的数字
            num_pattern = re.compile(r"_Part_(\d+)$")
            nums = []
            max_num = 0
            for name in names:
                match = num_pattern.search(name)
                if match:
                    num = int(match.group(1))
                    nums.append(num)
                    if num > max_num:
                        max_num = num

            # 生成 1 到 max_num 的顺序列表
            all_nums = list(range(1, max_num + 1))

            # 找出缺失的数字
            missing_numbers = list(set(all_nums) - set(nums))
            missing_numbers.sort()

    finally:
        driver.close()

    return missing_numbers

# if "__name__" == "__main__":
#     missing_numbers = find_missing_episodic_numbers(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
#     for number in missing_numbers:
#         print(number)
#     # print(missing_numbers)
result = find_missing_episodic_numbers(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
print("缺失的数字列表:", result)