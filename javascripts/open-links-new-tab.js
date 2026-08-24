document$.subscribe(function () {
  document.querySelectorAll("a[href]").forEach(function (a) {
    var href = a.getAttribute("href");
    if (!href || href.charAt(0) === "#" || href.indexOf("mailto:") === 0 || href.indexOf("tel:") === 0) return;
    a.target = "_blank";
    a.rel = "noopener";
  });
});
