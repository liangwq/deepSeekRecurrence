# core/tree.py

from anytree import NodeMixin, RenderTree

class ToTNode(NodeMixin):
    def __init__(self, name: str, value: float = 0.0, parent=None, children=None):
        super().__init__()
        self.name = name
        self.value = value
        self.parent = parent
        self.visits = 0
        
        # 强制 children 为列表，并确保父类使用此属性
        if children is None:
            self.__children = []
        else:
            self.__children = list(children)
        
    @property
    def children(self):
        """子节点列表（始终为 list）"""
        return self.__children
    
    @children.setter
    def children(self, value):
        """设置子节点时强制转换为列表"""
        self.__children = list(value)

class TreeManager:
    def __init__(self, root: ToTNode):
        self.root = root

    def print_tree(self, max_depth=3):
        print("\n=== 当前树结构 ===")
        for pre, _, node in RenderTree(self.root):
            if node.depth > max_depth:
                continue
            print(f"{pre}{node.name} (V={node.value:.2f}, N={node.visits})")

    def get_node_path(self, node: ToTNode) -> str:
        path = []
        current = node
        while current.parent:
            path.append(current.name)
            current = current.parent
        return " → ".join(reversed(path))