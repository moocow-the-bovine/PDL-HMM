# -*- Mode: Perl -*-
# t/03_baum.t: test Baum-Welch re-estimation

$TEST_DIR = './t';
#use lib qw(../blib/lib ../blib/arch); $TEST_DIR = '.'; # for debugging

# load common subs
use Test;
do "$TEST_DIR/common.plt";
use PDL;
use PDL::HMM;

BEGIN { plan tests=>9, todo=>[]; }

##-- test model 2:
sub testmodel2 {
  ##-- model
  $a = pdl(double, [[0,   1/2, 1/2],
		    [1/2,   0, 1/2],
		    [1/2, 1/2,   0]])->log;

  $b = pdl(double, [[1,   0,   0],
		    [0,   1,   0],
		    [0,   0,   1]])->log;

  $pi = pdl(double, [1,   0,   0])->log;

  ##-- inputs
  @os = ((map { pdl([0,1,2,0]) } (1..4)),
	 (map { pdl([0,2,1,0]) } (1..2)));
}

sub emreset {
  ##-- re-est: base
  ($ea,$eb,$epi) = hmmexpect0($a,$b,$pi);
  $etp           = logzero;
}

sub emE {
  foreach $o (@os) {
    ##-- alpha, beta
    $fw = hmmfw($a,$b,$pi,$o);
    $bw = hmmbw($a,$b,$o);

    ##-- text-prob
    $etp->inplace->logadd(logsumover($fw->slice("0,".($fw->dim(1)-1))));

    ##-- re-est: expect
    hmmexpect($a,$b,$pi, $o, $fw,$bw, $ea,$eb,$epi);
  }
  $etp -= log(scalar(@os));
}

sub emM {
  ##-- re-est: maximimize
  ($ahat,$bhat,$pihat) = hmmmaximize($ea,$eb,$epi);

  ##-- re-est: get new textprob
  $etphat = logzero;
  foreach $o (@os) {
    ##-- alpha
    $fw = hmmfw($ahat,$bhat,$pihat, $o);
    $etphat->inplace->logadd(logsumover($fw->slice(",".($fw->dim(1)-1))));
  }
  $etphat -= log(scalar(@os));

  ##-- now can compare text-probs: $etphat -- $etp
  $etpdiff = $etphat->logdiff($etp);
}

sub wantmodel2 {
  ##-- model 2: want: text-prob
  $etp_want = pdl(double, log(1/8));

  ##-- test: want: a
  $ea_want = pdl(double, [[0, 2, 4],
			  [4, 0, 2],
			  [2, 4, 0]])->log;

  $eb_want = pdl(double, [[6.75,    0,    0],
			  [   0, 0.75,    0],
			  [   0,    0, 0.75]])->log;

  $epi_want = pdl(double, [1,0,0])->log;

  ##-- model 2: want: etphat
  $etphat_want  = pdl(double, -1.5612357);
  $etpdiff_want = pdl(double, -2.4665575);

  ##-- model 2: want: maximized
  $ahat_want  = pdl(double, [[0,   1/3, 2/3],
			     [2/3,   0, 1/3],
			     [1/3, 2/3,   0]])->log;

  $bhat_want  = pdl(double, [[1, 0, 0],
			     [0, 1, 0],
			     [0, 0, 1]])->log;

  $pihat_want = pdl(double, [1, 0, 0])->log;
}

##-- tests: model 2

##-- 1--4: expect
testmodel2(); wantmodel2(); emreset(); emE(); emM();
isok("E(p(O))",     all($etp->approx($etp_want, 1e-6)));
isok("E(f(i-->j))", all($ea->approx($ea_want)));
isok("E(f(k @ j))", all($eb->approx($eb_want)));
isok("E(f(i @ 0))", all($epi->approx($epi_want)));

##-- 5--9: maximize
isok("E(^p(O))",   all($etphat_want->approx($etphat)));
isok("E(^p)-E(p)", all($etpdiff_want->approx($etpdiff)));
isok("Ahat",       all($ahat->approx($ahat_want)));
isok("Bhat",       all($bhat->approx($bhat_want)));
isok("pihat",      all($pihat->approx($pihat_want)));


print "\n";
# end of t/XX_yyyy.t

