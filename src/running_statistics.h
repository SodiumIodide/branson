#ifndef running_statistics_h_
#define running_statistics_h_

class Running_Statistics {
public:
  // constructor
  Running_Statistics() {
    m_n = 0;
    m_oldM = 0.0;
    m_newM = 0.0;
    m_oldS = 0.0;
    m_newS = 0.0;
    m_min = 0.0;
    m_max = 0.0;
  }

  void clear() {
    m_n = 0;
    m_oldM = 0.0;
    m_newM = 0.0;
    m_oldS = 0.0;
    m_newS = 0.0;
    m_min = 0.0;
    m_max = 0.0;
  }

  void push(double x) {
    m_n += 1;

    // See Knuth TAOCP vol 2, 3rd edition, page 232
    if (m_n == 1) {
      m_oldM = m_newM = x;
      m_oldS = 0.0;
      m_min = m_max = x;
    } else {
      m_newM = m_oldM + (x - m_oldM) / m_n;
      m_newS = m_oldS + (x - m_oldM) * (x - m_newM);

      // Set up for next iteration
      m_oldM = m_newM;
      m_oldS = m_newS;

      // Min and max
      if (x < m_min)
        m_min = x;
      else if (x > m_max)
        m_max = x;
    }
  }

  long num(void) { return m_n; }

  double mean(void) { return (m_n > 0) ? m_newM : 0.0; }

  double variance(void) { return (m_n > 1) ? (m_newS / (m_n - 1)) : 0.0; }

  double least(void) { return m_min; }

  double greatest(void) { return m_max; }

  // destructor
  ~Running_Statistics() { }

private:
  long m_n;
  double m_oldM;
  double m_newM;
  double m_oldS;
  double m_newS;
  double m_min;
  double m_max;
};

#endif
