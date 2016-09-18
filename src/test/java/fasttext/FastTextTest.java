package fasttext;

/**
 * Created by zeal on 16-9-18.
 */
public class FastTextTest {

    public static void main(String[] args) throws Exception {
        String fastTextModel = "./src/test/java/fasttext/model.bin";
        FastText fastText = new FastText(fastTextModel);
        String r = fastText.predictStr(fastTextModel, "游戏 界面 设计 设计师 工作 职责 1 、 负责 iosampandroid 手机 游戏 手机游戏 界面 设计 界面设计 的 整体 视觉 风格 2 、 手机 游戏 手机游戏 主 界面 和 功能 界面 ui 设计 3 、 手机 游戏 手机游戏 icon 、 logo 、 图标 等 游戏 元素 设计 4 、 界面 动画 效果 开发 5 、 与 项目 团队 配合 ， 参与 设计 体验 ， 对 产品 的 最终 ui 及 ue 效果 负责 6 、 面试 者 需带 相关 2d 设计 ui 游戏 作品 。 任职 要求 1 、 美术 设计 美术设计 专业 ， 专科 及 以上 上学 学历 以上学历 ， 具有 扎实 的 美术 基础 和 设计 功底 2 、 能 根据 策划 的 要求 ， 独立 完成 ui 、 界面 的 设计 ， 对于 整体 设计 能够 很 好 的 把握 3 、 热爱 游戏 行业 ， 有 很 好 的 职业 素养 ， 有 很 好 的 团队 合作 精神 ， 擅长 沟通 ， 能 承受 工作 压力 4 、 精通 ps 、 ai 、 flash 等 设计 工具 软件 工具软件 5 、 二年 以上 手游 设计 经验 ， 或有 成功 手游 设计 经验 者 优先 ");
        System.out.println(r);
    }
}
