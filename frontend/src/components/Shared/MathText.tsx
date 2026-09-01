import React, { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface MathTextProps {
  text: string;
  className?: string;
}

/**
 * Renders question text with inline LaTeX.
 * Detects \( ... \), $...$, and raw LaTeX commands like \sigma, \bowtie, \pi
 * and renders them via KaTeX.
 */
const MathText: React.FC<MathTextProps> = ({ text, className }) => {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!ref.current || !text) return;

    // Step 1: Normalize raw LaTeX commands that appear outside delimiters
    // Wrap standalone LaTeX commands in $...$ so KaTeX picks them up
    let processed = text;

    // Match patterns like \sigma_{cond}(R), \pi_{attrs}(R), \bowtie, etc.
    processed = processed.replace(
      /(\\(?:sigma|pi|bowtie|times|cup|cap|setminus|le|ge|ne|in|notin|subset|supset|forall|exists|sum|prod|int|infty|partial|nabla|cdot|ldots|cdots|alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|rho|tau|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega|mathbf|mathrm|mathcal|text|left|right|frac|sqrt|overline|underline|hat|tilde|bar|vec)(?:\{[^}]*\})*(?:\([^)]*\))*)/g,
      "$$$1$$"
    );

    // Step 2: Render with KaTeX
    try {
      // Replace \( ... \) and $...$ with KaTeX HTML
      const html = processed.replace(
        /\\\((.+?)\\\)|\$(.+?)\$/g,
        (_match, g1, g2) => {
          const tex = g1 || g2;
          try {
            return katex.renderToString(tex, {
              throwOnError: false,
              displayMode: false,
            });
          } catch {
            return `<code>${tex}</code>`;
          }
        }
      );
      ref.current.innerHTML = html;
    } catch {
      ref.current.textContent = text;
    }
  }, [text]);

  return <span ref={ref} className={className} />;
};

export default MathText;
