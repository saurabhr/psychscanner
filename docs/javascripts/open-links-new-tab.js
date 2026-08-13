document$.subscribe(function () {
  document.querySelectorAll('a[href="https://saurabhr.github.io/psychscanner-primal/"]').forEach(function (a) {
    a.target = "_blank";
    a.rel = "noopener";
  });
});
