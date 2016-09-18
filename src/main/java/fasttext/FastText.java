package fasttext;

import java.io.*;
import org.apache.log4j.Logger;

public class FastText implements Serializable {

    private static Logger logger = Logger.getLogger(FastText.class);
    Model model = null;

    public FastText() {
    }

    public FastText(String binFile) throws IOException {
        initModel(binFile);
    }

    public void initModel(String binFile)  throws IOException {
        model = new Model();
        model.loadModel(binFile);
        model.setTargetCounts();
    }

    public void predict(String binFile, String predictFile) throws IOException {
        if (model == null) {
            initModel(binFile);
        }
        model.predict(predictFile);
    }

    public String predictStr(String binFile, String str) throws IOException {
        if (model == null) {
            initModel(binFile);
        }
        return model.predictStr(str);
    }
}
