import uuid
from datetime import datetime
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from neo4j import AsyncGraphDatabase  # 假设使用Neo4j驱动

# 初始化数据库连接
async def get_driver():
    return AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")  # 替换为你的数据库凭据
    )

# 1. 创建实体节点 (Create)
async def create_entity_node(name, summary=None):
    driver = await get_driver()
    try:
        # 创建新节点
        node = EntityNode(
            uuid=str(uuid.uuid4()),
            name=name,
            summary=summary,
            created_at=datetime.now()
        )
        
        # 保存到数据库
        await node.save(driver)
        print(f"创建节点: {node.uuid} - {node.name}")
        return node
    finally:
        await driver.close()

# 2. 查询实体节点 (Read)
async def get_entity_node(node_uuid):
    driver = await get_driver()
    try:
        # 通过UUID查询节点
        node = await EntityNode.get_by_uuid(driver, node_uuid)
        if node:
            print(f"查询到节点: {node.uuid} - {node.name}")
            print(f"摘要: {node.summary}")
        return node
    finally:
        await driver.close()

# 3. 更新实体节点 (Update)
async def update_entity_node(node_uuid, new_name=None, new_summary=None):
    driver = await get_driver()
    try:
        # 先查询节点
        node = await EntityNode.get_by_uuid(driver, node_uuid)
        if not node:
            print("节点不存在")
            return
        # 更新属性
        if new_name:
            node.name = new_name
        if new_summary:
            node.summary = new_summary
        # 保存更新
        await node.save(driver)
        print(f"更新节点: {node.uuid} - {node.name}")
        return node
    finally:
        await driver.close()

# 4. 删除实体节点 (Delete)
async def delete_entity_node(node_uuid):
    driver = await get_driver()
    try:
        # 先查询节点
        node = await EntityNode.get_by_uuid(driver, node_uuid)
        if not node:
            print("节点不存在")
            return
        
        # 删除节点
        await node.delete(driver)
        print(f"删除节点: {node_uuid}")
    finally:
        await driver.close()

# 5. 创建实体关系 (Edge)
async def create_entity_relationship(source_uuid, target_uuid, relationship_name, fact):
    driver = await get_driver()
    try:
        # 创建边
        edge = EntityEdge(
            group_id="default",
            source_node_uuid=source_uuid,
            target_node_uuid=target_uuid,
            created_at=datetime.now(),
            name=relationship_name,
            fact=fact
        )
        
        # 保存边
        await edge.save(driver)
        print(f"创建关系: {source_uuid} -[{relationship_name}]-> {target_uuid}")
        return edge
    finally:
        await driver.close()

# 6. 完整示例流程
async def crud_demo():
    # 创建两个实体节点
    product = await create_entity_node(
        name="智能手表",
        summary="具有健康监测功能的智能手表"
    )
    
    category = await create_entity_node(
        name="可穿戴设备",
        summary="可穿戴在身上的电子设备类别"
    )
    
    # 创建节点间的关系
    if product and category:
        await create_entity_relationship(
            source_uuid=product.uuid,
            target_uuid=category.uuid,
            relationship_name="属于",
            fact="智能手表属于可穿戴设备类别"
        )
    
    # 查询节点详情
    if product:
        await get_entity_node(product.uuid)
    
    # 更新节点信息
    if product:
        await update_entity_node(
            node_uuid=product.uuid,
            new_summary="具有健康监测和GPS功能的智能手表"
        )
    
    # 清理演示数据
    if product:
        await delete_entity_node(product.uuid)
    if category:
        await delete_entity_node(category.uuid)

# 运行演示
if __name__ == "__main__":
    import asyncio
    asyncio.run(crud_demo())
