package spark.anomalydetection

object Main {

  def main(args: Array[String]): Unit = {
    import breeze.linalg._
    import breeze.plot._
    import breeze.numerics._

    val f = Figure()

    val p = f.subplot(0)
    val x = linspace(-10.0, 10.0)
    val z = sin(x)
    val plots = List(sin(x), cos(x))
    //    p += plot(x,sin(x) , '.')
    //    p += plot(x, cos(x), '.')
    plots.foreach(px => p += plot(x, px, '.'))
    p.title = "lines plotting"

    val p2 = f.subplot(2, 2, 1)
    val g2 = breeze.stats.distributions.Gaussian(0, 1)
    p2 += hist(g2.sample(100000), 100)
    p2.title = "A normal distribution"

    val p3 = f.subplot(2, 2, 2)
    val g3 = breeze.stats.distributions.Poisson(5)
    p3 += hist(g3.sample(100000), 100)
    p3.title = "A poisson distribution"

    val p4 = f.subplot(2, 2, 3)
    p4 += image(DenseMatrix.rand(200, 200))
    p4.title = "A random distribution"
    f.saveas("image.png")

  }

}