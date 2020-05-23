package water.rapids.ast.prims.advmath;

import hex.tfidf.DocumentFrequencyTask;
import hex.tfidf.InverseDocumentFrequencyTask;
import hex.tfidf.TermFrequencyTask;
import hex.tfidf.TfIdfPreprocessorTask;
import org.apache.log4j.Logger;
import water.Key;
import water.MRTask;
import water.Scope;
import water.fvec.*;
import water.rapids.Env;
import water.rapids.Merge;
import water.rapids.Rapids;
import water.rapids.Val;
import water.rapids.ast.AstPrimitive;
import water.rapids.ast.AstRoot;
import water.rapids.vals.ValFrame;

/**
 * Primitive AST operation to compute TF-IDF values for given document corpus.<br>
 * 
 * <br>
 * Input formats:
 * <p>
 * (Default) Row format when pre-processing is enabled: see {@link TfIdfPreprocessorTask}.
 * </p>
 * <p>
 * Row format when pre-processing is disabled: <code>documentID, word<code/>
 * </p>
 */
public class AstTfIdf extends AstPrimitive<AstTfIdf> {

    /**
     * Name to be used for a column containing Inverse Document Frequency values in the output frame of this operation.
     */
    private static final String IDF_COL_NAME = "IDF";
    /**
     * Name to be used for a column containing TF-IDF values in the output frame of this operation.
     */
    private static final String TF_IDF_COL_NAME = "TF-IDF";
    /**
     * Column names to be used for preprocessed frame.
     */
    private static final String[] PREPROCESSED_FRAME_COL_NAMES = new String[] { "DocID", "Word" };
    /**
     * Class logger.
     */
    private static Logger log = Logger.getLogger(AstTfIdf.class);

    @Override
    public int nargs() {
        return 1 + 2; // (tf-idf input_frame_name preprocess)
    }

    @Override
    public String[] args() {
        return new String[]{ "frame", "preprocess" };
    }

    @Override
    public Val apply(Env env, Env.StackHelp stk, AstRoot[] asts) {
        Frame inputFrame = stk.track(asts[1].exec(env).getFrame());
        boolean preprocess = asts[2].exec(env).getBool();
        
        if (inputFrame.anyVec().length() <= 0)
            throw new IllegalArgumentException("Empty input frame provided.");

        Scope.enter();
        Frame tfIdfFrame = null;
        try {
            // Pre-processing
            Frame wordFrame;
            long documentsCnt;
            if (preprocess) {
                byte[] outputTypes = new byte[]{Vec.T_NUM, Vec.T_STR};

                wordFrame = new TfIdfPreprocessorTask().doAll(outputTypes, inputFrame).outputFrame(PREPROCESSED_FRAME_COL_NAMES, null);
                documentsCnt = inputFrame.numRows();
            } else {
                if (inputFrame.numCols() < 2 || !inputFrame.vec(0).isNumeric() || !inputFrame.vec(1).isString())
                    throw new IllegalArgumentException("Incorrect format of a pre-processed input frame." +
                                                       "Following row format is expected: (numeric) documentID, (string) word.");

                wordFrame = inputFrame;
                String countDocumentsRapid = "(unique (cols " + asts[1].toString() + " [0]))";
                documentsCnt = Rapids.exec(countDocumentsRapid).getFrame().anyVec().length();
            }
            
            // DF
            Frame dfOutFrame = DocumentFrequencyTask.compute(wordFrame);
            Scope.track(dfOutFrame);

            // IDF
            InverseDocumentFrequencyTask idf = new InverseDocumentFrequencyTask(documentsCnt);
            Vec idfValues = idf.doAll(new byte[]{Vec.T_NUM}, dfOutFrame.lastVec()).outputFrame().anyVec();
            // Replace DF column with IDF column
            dfOutFrame.remove(dfOutFrame.numCols() - 1);
            dfOutFrame.add(IDF_COL_NAME, idfValues);

            // TF
            Frame tfOutFrame = TermFrequencyTask.compute(wordFrame);
            Scope.track(tfOutFrame);

            // Intermediate frame containing both TF and IDF values
            tfOutFrame.replace(1, tfOutFrame.vecs()[1].toCategoricalVec());
            dfOutFrame.replace(0, dfOutFrame.vecs()[0].toCategoricalVec());
            int[][] levelMaps = {
                    CategoricalWrappedVec.computeMap(tfOutFrame.vec(1).domain(), dfOutFrame.vec(0).domain())
            };
            Frame tfIdfIntermediate = Merge.merge(tfOutFrame, dfOutFrame, new int[]{1}, new int[]{0}, false, levelMaps);
            tfIdfIntermediate.replace(1, tfIdfIntermediate.vecs()[1].toStringVec());

            // TF-IDF
            int tfOutFrameColCnt = tfIdfIntermediate.numCols();
            TfIdfTask tfIdfTask = new TfIdfTask(tfOutFrameColCnt - 2, tfOutFrameColCnt - 1);
            Vec tfIdfValues = tfIdfTask.doAll(new byte[]{Vec.T_NUM}, tfIdfIntermediate).outputFrame().anyVec();
            Scope.track(tfIdfValues);

            // Construct final frame containing TF, IDF and TF-IDF values
            tfIdfIntermediate.add(TF_IDF_COL_NAME, tfIdfValues);
            tfIdfIntermediate._key = Key.make();

            if (log.isDebugEnabled())
                log.debug(tfIdfIntermediate.toTwoDimTable().toString());

            tfIdfFrame = tfIdfIntermediate;
        } finally {
            Key[] keysToKeep = tfIdfFrame != null ? tfIdfFrame.keys() : new Key[]{};
            Scope.exit(keysToKeep);
        }
        
        return new ValFrame(tfIdfFrame);
    }

    @Override
    public String str() {
        return "tf-idf";
    }

    /**
     * Final TF-IDF Map-Reduce task used to combine TF and IDF values together.
     */
    private static class TfIdfTask extends MRTask<TfIdfTask> {
        
        // IN
        /**
         * Index of a column containing Term Frequency values in the input frame of this task.
         */
        private final int _tfColIndex;
        /**
         * Index of a column containing Inverse Document Frequency values in the input frame of this task.
         */
        private final int _idfColIndex;
        
        private TfIdfTask(int tfColIndex, int idfColIndex) {
            _tfColIndex = tfColIndex;
            _idfColIndex = idfColIndex;
        }

        @Override
        public void map(Chunk[] cs, NewChunk nc) {
            Chunk tfValues = cs[_tfColIndex];
            Chunk idfValues = cs[_idfColIndex];
            
            for (int row = 0; row < tfValues._len; row++) {
                nc.addNum(tfValues.at8(row) * idfValues.atd(row));
            }
        }
    }
}
