import json
from pathlib import Path


LITERATURE = [
    {
        "title": "The eNanoMapper database for nanomaterial safety information",
        "year": 2015,
        "url": "https://www.beilstein-journals.org/bjnano/articles/6/165",
        "used_for": ["data_source", "schema_design"],
        "full_summary": "该文介绍 eNanoMapper 数据库和本体框架，重点解决纳米材料安全信息中材料标识、理化性质、实验条件、毒理终点和来源追溯难以统一的问题。论文对数据模板、ISA-TAB-Nano 兼容性、材料-实验-结果关系建模以及可检索接口进行了系统说明，为本项目把公开长表记录规范化为粒径、zeta、电荷、剂量、暴露时间、细胞类型、assay 和 viability 等机器学习字段提供了直接依据。",
        "method_relevance": "提供纳米材料安全数据的实体关系、字段归一化和可追溯来源设计思路。",
        "data_relevance": "eNanoMapper 是本任务公开清洗数据的优先来源，Solr 导出可保留原始 study/conditions/pchem 记录。",
        "limitations": "数据库聚合多项目数据，不同模板之间字段命名、单位和实验上下文差异较大，需要脚本化清洗和人工复核。",
        "key_points": [
            "纳米材料安全数据需要同时保留材料、实验条件和毒性终点。",
            "本体和模板有助于跨项目合并数据。",
            "公开接口适合第一版模型训练数据的可追溯导出。",
        ],
    },
    {
        "title": "Predicting Cytotoxicity of Nanoparticles: A Meta-Analysis Using Machine Learning",
        "year": 2024,
        "url": "https://doi.org/10.1021/acsanm.4c02269",
        "used_for": ["model_design", "feature_selection", "feature_explanation"],
        "full_summary": "该研究围绕纳米颗粒细胞毒性预测进行机器学习元分析，强调粒径、表面电荷、材料组成、剂量、暴露时间、细胞类型和 assay 条件在跨文献建模中的核心作用。论文比较了多种模型和特征贡献方式，说明细胞毒性预测不能只依赖材料名称，还需要把实验条件作为模型输入。本项目采用这些字段作为第一版 schema，并在解释报告中披露特征和文献依据。",
        "method_relevance": "支撑使用表格特征、监督学习和特征重要性解释来预测 cell viability。",
        "data_relevance": "为本项目选择粒径、zeta、剂量、暴露时间、材料类型、细胞类型和 assay 提供依据。",
        "limitations": "元分析受原文献异质性、单位换算、细胞系差异和发表偏倚影响，模型结果不能直接替代实验验证。",
        "key_points": [
            "理化性质与实验条件需要共同建模。",
            "cell viability 可作为连续回归目标，也可派生毒性等级。",
            "跨文献数据清洗质量直接影响预测可信度。",
        ],
    },
    {
        "title": "Application of Machine Learning in Nanotoxicology: A Critical Review and Perspective",
        "year": 2024,
        "url": "https://doi.org/10.1021/acs.est.4c03217",
        "used_for": ["background", "limitations"],
        "full_summary": "该综述从纳米毒理学的机器学习应用出发，讨论数据稀疏、实验标准不一致、外推能力有限、模型可解释性不足和数据共享不足等关键问题。它强调纳米毒性模型必须记录数据来源、清洗规则、适用域和不确定性，不能把小型或合成样例当作训练依据。本项目的 metadata、来源披露和公开数据下限正是针对这些风险设置。",
        "method_relevance": "提醒模型开发需关注适用域、验证策略和可解释性，而不仅是单次指标。",
        "data_relevance": "支持在清洗结果中保留来源、记录数量、排除策略和科学使用限制。",
        "limitations": "综述提供的是方法论与风险框架，不提供可直接训练的统一表格数据。",
        "key_points": [
            "纳米毒理数据高度异质，清洗过程必须可审计。",
            "机器学习预测需要说明适用范围和局限。",
            "公开数据共享是模型可复现的前提。",
        ],
    },
    {
        "title": "caNanoLab cancer nanotechnology data portal",
        "year": 2013,
        "url": "https://cananolab.cancer.gov/",
        "used_for": ["cancer_nanomedicine_context", "deferred_source_cross_check"],
        "full_summary": "caNanoLab 面向癌症纳米技术和生物医学纳米材料数据共享，覆盖纳米材料表征、体外与体内实验、样品制备以及与癌症应用相关的上下文信息。当前阶段用户已决定不把 caNanoLab 接入训练清洗流水线，因此该资源只用于肿瘤纳米药物筛选背景和后续来源交叉核验，不暗示本版 toxicity_clean.csv 已合并 caNanoLab 记录。",
        "method_relevance": "帮助定义智能体报告中的癌症纳米医学场景和实验建议边界。",
        "data_relevance": "作为后续交叉校验和癌症纳米药物上下文数据源保留在优先清单中；本阶段训练数据不接入 caNanoLab CSV。",
        "limitations": "平台页面和数据入口并不总是直接提供与本 schema 完全一致的批量 CSV，需要后续单独清洗。",
        "key_points": [
            "癌症纳米技术数据需要材料表征和生物实验共同描述。",
            "数据共享可加速候选纳米药物评估。",
            "本项目需要把毒性预测放在生物医学应用上下文中披露。",
        ],
    },
    {
        "title": "AI and Machine Learning Approaches for Predicting Nanoparticles Toxicity: The Critical Role of Physiochemical Properties",
        "year": 2024,
        "url": "https://arxiv.org/abs/2409.15322",
        "used_for": ["feature_selection", "feature_explanation"],
        "full_summary": "该预印本强调纳米颗粒毒性预测中理化性质的核心地位，尤其是粒径、表面积、形貌、表面电荷、组成和表面修饰如何影响细胞摄取、膜相互作用、氧化应激和炎症反应。它为本项目的特征解释提供了生物机制语言：同样剂量下，不同粒径或 zeta 可能改变细胞暴露和毒性响应，因此解释模块应同时说明特征方向和科学不确定性。",
        "method_relevance": "支持把理化性质作为模型输入和解释报告中的重点因子。",
        "data_relevance": "强化粒径、zeta、材料类型和表面修饰在训练数据 schema 中的必要性。",
        "limitations": "预印本结论需结合同行评议文献和具体实验体系使用，不能单独作为定量阈值来源。",
        "key_points": [
            "理化性质是纳米毒性机器学习的关键解释维度。",
            "表面电荷和粒径影响细胞相互作用。",
            "特征解释需要连接统计重要性与毒理机制。",
        ],
    },
    {
        "title": "Smart Drug-Delivery Systems for Cancer Nanotherapy",
        "year": 2024,
        "url": "https://arxiv.org/abs/2401.11192",
        "used_for": ["cancer_nanomedicine_context"],
        "full_summary": "该预印本综述智能药物递送系统在癌症纳米治疗中的设计方向，包括刺激响应材料、靶向递送、控释、肿瘤微环境响应和联合治疗策略。它并不是细胞毒性表格数据源，但能帮助本项目把纳米材料安全预测放到肿瘤治疗候选物筛选语境中：材料需要在疗效、递送效率、生物相容性和毒性之间权衡。",
        "method_relevance": "用于报告生成和应用场景说明，帮助解释为什么需要毒性预测和实验建议。",
        "data_relevance": "不直接提供训练数据，但支撑癌症纳米药物筛选背景和风险披露。",
        "limitations": "主题偏药物递送系统综述，不能替代细胞毒性建模文献或公开训练数据。",
        "key_points": [
            "癌症纳米治疗关注靶向递送和控释。",
            "安全性与疗效需要共同评估。",
            "毒性预测适合作为早期筛选和实验设计辅助。",
        ],
    },
]


def main() -> None:
    Path("literature").mkdir(parents=True, exist_ok=True)
    Path("literature/literature_base.json").write_text(
        json.dumps(LITERATURE, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
